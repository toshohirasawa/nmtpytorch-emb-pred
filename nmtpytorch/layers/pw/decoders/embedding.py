# -*- coding: utf-8 -*-
from collections import defaultdict
import random
import math

import torch
from torch import nn
import torch.nn.functional as F

from ....utils.nn import get_rnn_hidden_state
from ....utils.functions import log_vmf_c, log_vmf_c_approx
from ... import FF
from ...attention import get_attention
from ..embedding import EmbeddingOutput
from ..max_margin import MaxMarginLoss

class EmbeddingOutputDecoder(nn.Module):
    """A conditional decoder with attention à la dl4mt-tutorial."""
    def __init__(self, input_size, hidden_size, ctx_size_dict, ctx_name, n_vocab,
                 rnn_type, num_layer=1, tied_emb=False, dec_init='zero', dec_init_activ='tanh',
                 dec_init_size=None, att_type='mlp',
                 att_activ='tanh', att_bottleneck='ctx', att_temp=1.0,
                 transform_ctx=True, mlp_bias=False, dropout_out=0,
                 emb_maxnorm=None, emb_gradscale=False, sched_sample=0,
                 bos_type='emb', bos_dim=None, bos_activ=None, bos_bias=False,
                 loss_type='maxmargin', loss_args={}):
        super().__init__()

        # Normalize case
        self.rnn_type = rnn_type.upper()
        self.loss_type = loss_type.lower()

        # Safety checks
        assert self.rnn_type in ('GRU', 'LSTM'), \
            "rnn_type '{}' not known".format(rnn_type)
        assert bos_type in ('emb', 'feats', 'zero'), "Unknown bos_type"
        assert dec_init.startswith(('zero', 'feats', 'mean_ctx', 'max_ctx', 'last_ctx')), \
            "dec_init '{}' not known".format(dec_init)

        assert self.loss_type in ('cossim', 'maxmargin'), \
            f"unknown loss_type '{self.loss_type}' not know"

        RNN = getattr(nn, '{}Cell'.format(self.rnn_type))
        # LSTMs have also the cell state
        self.n_states = 1 if self.rnn_type == 'GRU' else 2

        # Set custom handlers for GRU/LSTM
        if self.rnn_type == 'GRU':
            self._rnn_unpack_states = lambda x: x
            self._rnn_pack_states = lambda x: x
        elif self.rnn_type == 'LSTM':
            self._rnn_unpack_states = self._lstm_unpack_states
            self._rnn_pack_states = self._lstm_pack_states

        # Set decoder initializer
        self._init_func = getattr(self, '_rnn_init_{}'.format(dec_init))

        # Other arguments
        self.num_layer = num_layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ctx_size_dict = ctx_size_dict
        self.ctx_name = ctx_name
        self.n_vocab = n_vocab
        self.tied_emb = tied_emb
        self.dec_init = dec_init
        self.dec_init_size = dec_init_size
        self.dec_init_activ = dec_init_activ
        self.att_bottleneck = att_bottleneck
        self.att_activ = att_activ
        self.att_type = att_type
        self.att_temp = att_temp
        self.transform_ctx = transform_ctx
        self.mlp_bias = mlp_bias
        self.dropout_out = dropout_out
        self.emb_maxnorm = emb_maxnorm
        self.emb_gradscale = emb_gradscale
        self.sched_sample = sched_sample
        self.bos_type = bos_type
        self.bos_dim = bos_dim
        self.bos_activ = bos_activ
        self.bos_bias = bos_bias
        self.loss_args = loss_args

        if self.bos_type == 'feats':
            # Learn a <bos> embedding
            self.ff_bos = FF(self.bos_dim, self.input_size, bias=self.bos_bias,
                             activ=self.bos_activ)

        # Create target embeddings
        self.emb = nn.Embedding(self.n_vocab, self.input_size,
            padding_idx=0, max_norm=self.emb_maxnorm,
            scale_grad_by_freq=self.emb_gradscale)

        if self.dec_init != 'zero':
            # For source-based inits, input size is the encoding size
            # For 'feats', it's given by dec_init_size, no need to infer
            if self.dec_init.endswith('_ctx'):
                self.dec_init_size = self.ctx_size_dict[self.ctx_name]
            # Add a FF layer for decoder initialization
            self.ff_dec_init = FF(
                self.dec_init_size,
                self.hidden_size * self.n_states,
                activ=self.dec_init_activ)

        # Create decoders
        self.dec0 = RNN(self.input_size, self.hidden_size)

        Attention = get_attention(self.att_type)
        for l in range(1, self.num_layer+1):
            # Create attention layer
            att = Attention(
                self.ctx_size_dict[self.ctx_name],
                self.hidden_size,
                transform_ctx=self.transform_ctx,
                mlp_bias=self.mlp_bias,
                att_activ=self.att_activ,
                att_bottleneck=self.att_bottleneck,
                temp=self.att_temp)
            setattr(self, f'att{l}', att)
            setattr(self, f'dec{l}', RNN(self.hidden_size, self.hidden_size))

        # Output dropout
        if self.dropout_out > 0:
            self.do_out = nn.Dropout(p=self.dropout_out)

        # Output bottleneck: maps hidden states to target emb dim
        self.hid2out = FF(self.hidden_size, self.input_size,
                          bias_zero=True, activ='tanh')

        # Final similarity with embeddings
        self.out2prob = EmbeddingOutput(self.n_vocab, self.input_size)

        # Loss function (cossim, maxmargin, vMF=von Mises-Fisher)
        # self.nll_loss = nn.NLLLoss(reduction="sum", ignore_index=0)
        self.prob2loss = getattr(self, '_loss_' + self.loss_type)()

    def _loss_cossim(self):
        return lambda log_p, y: torch.nn.functional.nll_loss(log_p - 1, y, reduction='sum', ignore_index=0)

    def _loss_maxmargin(self):
        for kw in ['margin', 'constrastive_type']:
            assert kw in self.loss_args, f"'{kw}' should be given"
        return MaxMarginLoss(margin=self.loss_args['margin'], constrastive_type=self.loss_args['constrastive_type'])

    def _lstm_pack_states(self, h):
        """Pack LSTM hidden and cell state."""
        return torch.cat(h, dim=-1)

    def _lstm_unpack_states(self, h):
        """Unpack LSTM hidden and cell state to tuple."""
        return torch.split(h, self.hidden_size, dim=-1)

    def _rnn_init_zero(self, ctx_dict):
        """Zero initialization."""
        ctx, _ = ctx_dict[self.ctx_name]
        return torch.zeros(
            ctx.shape[1], self.hidden_size * self.n_states, device=ctx.device)

    def _rnn_init_mean_ctx(self, ctx_dict):
        """Initialization with mean-pooled source annotations."""
        ctx, ctx_mask = ctx_dict[self.ctx_name]
        return self.ff_dec_init(
            ctx.sum(0).div(ctx_mask.unsqueeze(-1).sum(0))
            if ctx_mask is not None else ctx.mean(0))

    def _rnn_init_max_ctx(self, ctx_dict):
        """Initialization with max-pooled source annotations."""
        ctx, ctx_mask = ctx_dict[self.ctx_name]
        # Max-pooling may not care about mask (depends on non-linearity maybe)
        return self.ff_dec_init(ctx.max(0)[0])

    def _rnn_init_last_ctx(self, ctx_dict):
        """Initialization with the last source annotation."""
        ctx, ctx_mask = ctx_dict[self.ctx_name]
        if ctx_mask is None:
            h_0 = self.ff_dec_init(ctx[-1])
        else:
            last_idxs = ctx_mask.sum(0).sub(1).long()
            h_0 = self.ff_dec_init(ctx[last_idxs, range(ctx.shape[1])])
        return h_0

    def _rnn_init_feats(self, ctx_dict):
        """Feature based decoder initialization."""
        return self.ff_dec_init(ctx_dict[self.aux_ctx_name][0][0])

    def get_emb(self, idxs, tstep):
        """Returns time-step based embeddings."""
        if tstep == 0:
            if self.bos_type == 'emb':
                # Learned <bos> embedding
                return self.emb(idxs)
            elif self.bos_type == 'zero':
                # Constant-zero <bos> embedding
                return torch.zeros(
                    idxs.shape[0], self.input_size, device=idxs.device)
            else:
                # Feature-based <bos> computed in f_init()
                return self.bos
        # For other timesteps, look up the embedding layer
        return self.emb(idxs)

    def f_init(self, ctx_dict):
        """Returns the initial h_0 for the decoder."""
        self.history = defaultdict(list)
        # Compute <bos> out of 'feats' if requested
        if self.bos_type == 'feats':
            self.bos = self.ff_bos(ctx_dict['feats'][0])
        return self._init_func(ctx_dict)

    def f_next(self, ctx_dict, y, h):
        """Applies one timestep of recurrence."""
        # Get hidden states from the first decoder (purely cond. on LM)
        h1_c1 = self.dec0(y, self._rnn_unpack_states(h))
        h1 = get_rnn_hidden_state(h1_c1)

        h_c, h = h1_c1, h1
        for l in range(1, self.num_layer+1):
            # apply attention
            att = getattr(self, f'att{l}')
            txt_alpha_t, txt_z_t = att(
                h.unsqueeze(0), *ctx_dict[self.ctx_name]
            )
            if not self.training:
                self.history[f'alpha_txt_{l}'].append(txt_alpha_t)
            
            # Run second decoder (h1 is compatible now as it was returned by GRU)
            dec = getattr(self, f'dec{l}')
            h_c = dec(txt_z_t, h_c)
            h = get_rnn_hidden_state(h_c)


        # This is a bottleneck to avoid going from H to V directly
        logit = self.hid2out(h)

        # conver to unit vector for easy cossim calculation
        logit = logit / logit.norm(dim=-1, keepdim=True)

        # Apply dropout if any
        if self.dropout_out > 0:
            logit = self.do_out(logit)

        # Transform logit to T*B*V (V: vocab_size)
        # Compute similarity over token embeddings
        log_p = self.out2prob(logit)

        # Return log probs and new hidden states
        return log_p, self._rnn_pack_states(h_c)

    def forward(self, ctx_dict, y):
        """Computes the softmax outputs given source annotations `ctx_dict[self.ctx_name]`
        and ground-truth target token indices `y`. Only called during training.

        Arguments:
            ctx_dict(dict): A dictionary of tensors that should at least contain
                the key `ctx_name` as the main source representation of shape
                S*B*ctx_dim`.
            y(Tensor): A tensor of `T*B` containing ground-truth target
                token indices for the given batch.
        """

        loss = 0.0

        # Get initial hidden state
        h = self.f_init(ctx_dict)

        # are we doing scheduled sampling?
        sched = self.training and (random.random() > (1 - self.sched_sample))

        # Convert token indices to embeddings -> T*B*E
        # Skip <bos> now
        bos = self.get_emb(y[0], 0)
        log_p, h = self.f_next(ctx_dict, bos, h)
        loss += self.prob2loss(log_p, y[1])
        y_emb = self.emb(y[1:])

        for t in range(y_emb.shape[0] - 1):
            emb = self.emb(log_p.argmax(1)) if sched else y_emb[t]
            log_p, h = self.f_next(ctx_dict, emb, h)
            loss += self.prob2loss(log_p, y[t + 2])

        return {'loss': loss}
