# -*- coding: utf-8 -*-
import logging

import torch
from torch import nn

import numpy as np

from ...layers import TextEncoder
from ...layers.decoders import get_decoder
from ...layers import PWImaginationDecoder, PWEmbeddingOutputDecoder
from ...layers import ZSpace
from ...utils.misc import get_n_params
from ...vocabulary import Vocabulary
from ...utils.topology import Topology
from ...utils.ml_metrics import Loss
from ...utils.device import DEVICE
from ...utils.misc import pbar
from ...utils.embedding import load_fasttext
from ...datasets import MultimodalDataset
from ...metrics import Metric
from ...utils.scheduler import Scheduler

logger = logging.getLogger('nmtpytorch')


class Imagination(nn.Module):
    supports_beam_search = True

    def set_defaults(self):
        self.defaults = {
            # Text related options
            'emb_dim': 128,             # Source and target embedding sizes
            'emb_maxnorm': None,        # Normalize embeddings l2 norm to 1
            'emb_gradscale': False,     # Scale embedding gradients w.r.t. batch frequency
            'enc_dim': 256,             # Encoder hidden size
            'enc_type': 'gru',          # Encoder type (gru|lstm)
            'enc_lnorm': False,         # Add layer-normalization to encoder output
            'enc_emb_init_type': 'random',
            'enc_emb_init_args': {},
            'n_encoders': 1,            # Number of stacked encoders
            'dec_dim': 256,             # Decoder hidden size
            'dec_type': 'gru',          # Decoder type (gru|lstm)
            'dec_variant': 'cond',
            'dec_init': 'mean_ctx',     # How to initialize decoder (zero/mean_ctx/feats)
            'dec_init_size': None,      # feature vector dimensionality for
            'dec_init_activ': 'tanh',   # Decoder initialization activation func
                                        # dec_init == 'feats'
            'dec_emb_init_type': 'random',
            'dec_emb_init_args': {},
            'dec_out_init_type': 'fasttext',
            'dec_out_init_args': {},
            'n_decoders': 1,
            'loss_type': 'maxmargin',   # Loss type for decoder
            'loss_args': {},
            'att_type': 'mlp',          # Attention type (mlp|dot)
            'att_temp': 1.,             # Attention temperature
            'att_activ': 'tanh',        # Attention non-linearity (all torch nonlins)
            'att_mlp_bias': False,      # Enables bias in attention mechanism
            'att_bottleneck': 'ctx',    # Bottleneck dimensionality (ctx|hid)
            'att_transform_ctx': True,  # Transform annotations before attention
            'dropout_emb': 0,           # Simple dropout to source embeddings
            'dropout_ctx': 0,           # Simple dropout to source encodings
            'dropout_out': 0,           # Simple dropout to decoder output
            'dropout_enc': 0,           # Intra-encoder dropout if n_encoders > 1
            'tied_emb': False,          # Share embeddings: (False|2way|3way)
            'direction': None,          # Network directionality, i.e. en->de
            'max_len': 80,              # Reject sentences where 'bucket_by' length > 80
            'bucket_by': None,          # A key like 'en' to define w.r.t which dataset
                                        # the batches will be sorted
            'bucket_order': None,       #
            'sampler_type': 'bucket',   # bucket or approximate
            'sched_sampling': 0,        # Scheduled sampling ratio
            'bos_type': 'emb',          # 'emb': default learned emb
            'bos_activ': None,          #
            'bos_dim': None,            #

            # # Image related options
            'n_channels': 2048,         # depends on the features used

            # Latent space related options
            'mtl_loss_type': 'naive',   # loss handling
            'z_w': 0.50,                # training objective interpolation parameter
            'aux_cap': -1,              # cap of times to learn from auxiliary task
        }

    def __init__(self, opts):
        super().__init__()

        # opts -> config file sections {.model, .data, .vocabulary, .train}
        self.opts = opts

        # Setup options
        self.opts.model = self.set_model_options(opts.model)

        # Each auxiliary loss should be stored inside this dictionary
        # in order to be taken into account by the mainloop for multi-tasking
        self.aux_loss = {}

        # Vocabulary
        self.vocabs = {}
        for lang, fname in self.opts.vocabulary.items():
            self.vocabs[lang] = Vocabulary(fname, name=lang)
        
        # Parse topology & languages
        self.topology = Topology(self.opts.model['direction'])
        self.sl = self.topology.get_src_langs()[0]
        self.src_vocab = self.vocabs[self.sl]
        self.n_src_vocab = len(self.src_vocab)
        self.tl = self.topology.get_trg_langs()[0]
        self.trg_vocab = self.vocabs[self.tl]
        self.n_trg_vocab = len(self.trg_vocab)
        self.mtl_loss_type = self.opts.model['mtl_loss_type'].lower()

        # references
        # TODO: valuation data sould be excluded from model
        self.val_refs = self.opts.data['val_set'][self.tl]


        # MT model parameters

        # Textual context size
        # it should be (enc_dim * 2) as it is the concat of forward and backward
        if 'enc_dim' in self.opts.model:
            self.ctx_sizes = {str(self.sl): self.opts.model['enc_dim'] * 2}


        # Check tying option
        if self.opts.model['tied_emb'] not in [False, '2way']:
            raise RuntimeError(
                "'{}' not recognized for tied_emb.".format(self.opts.model['tied_emb']))

        # IMAGINATION model parameters
        self.ctx_sizes['image'] = self.opts.model['n_channels']

        # MTL parameters
        assert self.mtl_loss_type in ('naive', 'mixed'), \
            f"Unknown mtl_loss_type (value: '{self.mtl_loss_type}')."

        self.forward = getattr(self, f'_forward_{self.mtl_loss_type}')
        self.z_w = self.opts.model['z_w']
        if self.mtl_loss_type == 'naive':
            # As z_w is a interpolation coefficient,
            # convertion to propability of aux task execution
            # self.z_w = self.z_w / (1. + self.z_w)
            if self.z_w >= 0.5:
                self.z_w = 1
            else:
                self.z_w = self.z_w / (1 - self.z_w)
        # handle times of auxiliary task learning
        self.aux_cap = self.opts.model['aux_cap']
        self.n_aux_learning = 0
        if self.aux_cap < 0:
            self.run_aux_task = lambda: True
        else:
            self.run_aux_task = lambda: self.n_aux_learning < self.aux_cap 

    def __repr__(self):
        s = super().__repr__() + '\n'
        for vocab in self.vocabs.values():
            s += "{}\n".format(vocab)
        s += "{}\n".format(get_n_params(self))
        return s

    def set_model_options(self, model_opts):
        self.set_defaults()
        for opt, value in model_opts.items():
            if opt in self.defaults:
                # Override defaults from config
                if isinstance(self.defaults[opt], dict):
                    # translation or test
                    if isinstance(value, dict):
                        self.defaults[opt].update(value)
                    # train
                    else:
                        self.defaults[opt].update(getattr(self.opts, str(value), {}))
                else:
                    self.defaults[opt] = value
            else:
                logger.info('Warning: unused model option: {}'.format(opt))
        return self.defaults

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad and 'bias' not in name:
                nn.init.kaiming_normal_(param.data)
        
        # initialize embedding and output layers
        if self.opts.model['enc_emb_init_type'] == 'fasttext':
            args = self.opts.model['enc_emb_init_args']
            self.enc.emb.weight = self.emb_from_fasttext(self.src_vocab, args)

        if self.opts.model['dec_emb_init_type'] == 'fasttext':
            args = self.opts.model['dec_emb_init_args']
            self.dec.emb.weight = self.emb_from_fasttext(self.trg_vocab, args)
        
        # reset decoder specific parameters
        getattr(self, f'_reset_dec_param_{self.dec_variant}', lambda : None)()

    def _reset_dec_param_embedding_output(self):
        self.dec_out_init_type = self.opts.model['dec_out_init_type'].lower()
        if self.dec_out_init_type == 'fasttext':
            args = self.opts.model['dec_out_init_args']
            weight = self.emb_from_fasttext(self.trg_vocab, args)
            weight.requires_grad = False
            self.dec.out2prob.weight = weight
        elif self.dec_out_init_type == 'tied':
            self.dec.out2prob.weight = self.dec.emb.weight

    def emb_from_fasttext(self, vocab, args):
        bin_fname = args['pretrained_file']
        center_type = args['center_type']
        sub_pca = args['sub_pca']
        n_pca = args['n_pca']
        emb = load_fasttext(vocab=vocab, bin_fname=bin_fname,
            normalize=True, center_type=center_type, sub_pca=sub_pca, pca_size=n_pca).cuda()
        return torch.nn.Parameter(emb)

    def setup(self, is_train=True):
        self.opts.model['enc_emb_init_args']['vocab'] = self.src_vocab
        # Shared encoder
        self.enc = TextEncoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['enc_dim'],
            n_vocab=self.n_src_vocab,
            rnn_type=self.opts.model['enc_type'],
            dropout_emb=self.opts.model['dropout_emb'],
            dropout_ctx=self.opts.model['dropout_ctx'],
            dropout_rnn=self.opts.model['dropout_enc'],
            num_layers=self.opts.model['n_encoders'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            layer_norm=self.opts.model['enc_lnorm'])

        # MT decoder
        self.dec_variant = self.opts.model['dec_variant'].lower()
        dec_inits = {
            'embedding_output': self._dec_embedding_output,
        }
        if self.dec_variant in dec_inits:
            self.dec = dec_inits[self.dec_variant]()
        else:
            Decoder = get_decoder(self.opts.model['dec_variant'])
            self.dec = Decoder(
                input_size=self.opts.model['emb_dim'],
                hidden_size=self.opts.model['dec_dim'],
                n_vocab=self.n_trg_vocab,
                rnn_type=self.opts.model['dec_type'],
                ctx_size_dict=self.ctx_sizes,
                ctx_name=str(self.sl),
                tied_emb=self.opts.model['tied_emb'],
                dec_init=self.opts.model['dec_init'],
                dec_init_size=self.opts.model['dec_init_size'],
                dec_init_activ=self.opts.model['dec_init_activ'],
                att_type=self.opts.model['att_type'],
                att_temp=self.opts.model['att_temp'],
                att_activ=self.opts.model['att_activ'],
                transform_ctx=self.opts.model['att_transform_ctx'],
                mlp_bias=self.opts.model['att_mlp_bias'],
                att_bottleneck=self.opts.model['att_bottleneck'],
                dropout_out=self.opts.model['dropout_out'],
                emb_maxnorm=self.opts.model['emb_maxnorm'],
                emb_gradscale=self.opts.model['emb_gradscale'],
                sched_sample=self.opts.model['sched_sampling'],
                bos_type=self.opts.model['bos_type'],
                bos_dim=self.opts.model['bos_dim'],
                bos_activ=self.opts.model['bos_activ'])

        # IMAGINATION decoder
        self.img_dec = PWImaginationDecoder(
            ctx_size=self.opts.model['dec_dim'] * 2,    # bidirectional
            output_size=self.ctx_sizes['image'],
            att_activ=self.opts.model['att_activ']
        )

    def _dec_embedding_output(self):
        self.opts.model['dec_emb_init_args']['vocab'] = self.trg_vocab
        self.opts.model['dec_out_init_args']['vocab'] = self.trg_vocab
        return PWEmbeddingOutputDecoder(
            input_size=self.opts.model['emb_dim'],
            hidden_size=self.opts.model['dec_dim'],
            n_vocab=self.n_trg_vocab,
            rnn_type=self.opts.model['dec_type'],
            num_layer=self.opts.model['n_decoders'],
            ctx_size_dict=self.ctx_sizes,
            ctx_name=str(self.sl),
            tied_emb=self.opts.model['tied_emb'],
            dec_init=self.opts.model['dec_init'],
            dec_init_size=self.opts.model['dec_init_size'],
            dec_init_activ=self.opts.model['dec_init_activ'],
            loss_type=self.opts.model['loss_type'],
            loss_args=self.opts.model['loss_args'],
            att_type=self.opts.model['att_type'],
            att_temp=self.opts.model['att_temp'],
            att_activ=self.opts.model['att_activ'],
            transform_ctx=self.opts.model['att_transform_ctx'],
            mlp_bias=self.opts.model['att_mlp_bias'],
            att_bottleneck=self.opts.model['att_bottleneck'],
            dropout_out=self.opts.model['dropout_out'],
            emb_maxnorm=self.opts.model['emb_maxnorm'],
            emb_gradscale=self.opts.model['emb_gradscale'],
            sched_sample=self.opts.model['sched_sampling'],
            bos_type=self.opts.model['bos_type'],
            bos_dim=self.opts.model['bos_dim'],
            bos_activ=self.opts.model['bos_activ'])

    def load_data(self, split, batch_size, mode='train'):
        """Loads the requested dataset split."""
        dataset = MultimodalDataset(
            data=self.opts.data['{}_set'.format(split)],
            mode=mode, batch_size=batch_size,
            vocabs=self.vocabs, topology=self.topology,
            bucket_by=self.opts.model['bucket_by'],
            max_len=self.opts.model['max_len'],
            bucket_order=self.opts.model['bucket_order'])
        logger.info(dataset)
        return dataset

    def get_bos(self, batch_size):
        return torch.LongTensor(batch_size).fill_(self.trg_vocab['<bos>'])

    def encode(self, batch, **kwargs):
        d = {
            str(self.sl): self.enc(batch[self.sl]),
            'image': (batch['image'], None)
        }
        return d

    def _forward_mixed(self, batch, is_train=True, **kwargs):
        """
        Learn parameter jointly.
        """
        result = self.dec(self.encode(batch), batch[self.tl])
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]

        if is_train and self.run_aux_task():
            loss = result['loss'] / result['n_items']

            aux_result = self.img_dec(self.encode(batch)[self.sl], batch['image'])
            aux_loss = aux_result['loss']

            result = {
                'loss': self.z_w * loss + (1 - self.z_w) * aux_loss,
                'n_items': 1
            }

            self.n_aux_learning += 1

        return result

    def _forward_naive(self, batch, is_train=True, **kwargs):
        """
        Learn parameters separately.
        Similar implementation in IMAGINATION (Elliot et al., 2017)
        """
        if is_train  and self.run_aux_task() and (np.random.rand() < self.z_w):
            img_result = self.img_dec(self.encode(batch)[self.sl], batch['image'])
            loss = img_result['loss']
            loss.backward()
            self.n_aux_learning += 1

        result = self.dec(self.encode(batch), batch[self.tl])
        result['n_items'] = torch.nonzero(batch[self.tl][1:]).shape[0]
        return result

    def test_performance(self, data_loader, dump_file=None):
        """Computes test set loss over the given DataLoader instance."""
        loss = Loss()

        for batch in pbar(data_loader, unit='batch'):
            batch.device(DEVICE)
            out = self.forward(batch, is_train=False)
            loss.update(out['loss'], out['n_items'])

        return [
            Metric('LOSS', loss.get(), higher_better=False),
        ]

    def get_decoder(self, task_id=None):
        return self.dec
