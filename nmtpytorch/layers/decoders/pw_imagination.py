# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn

from ...utils.nn import get_activation_fn

class PWImaginationDecoder(nn.Module):
    """
    Elliott, Kádár - 2017 - Imagination improves Multimodal Translation
    """
    def __init__(self, ctx_size, output_size=2048, 
                 att_activ='tanh',
                 margin=0.1, n_negatives=1):
        super().__init__()

        self.ctx_size = ctx_size
        self.output_size = output_size
        self.margin = margin
        self.n_negatives = n_negatives

        self.hid2out = nn.Linear(self.ctx_size, self.output_size,
                                 bias=False)
        self.activ = get_activation_fn(att_activ)

    def forward(self, ctx, feats):
        # predict image feats (size: TxBxS)
        ctx_, mask = ctx
        n_tokens = mask.sum(dim=0).type(ctx_.dtype)
        mean_ctx = ctx_.sum(dim=0) / n_tokens.unsqueeze(-1)
        out = self.hid2out(mean_ctx)
        out = self.activ(out)

        B = out.shape[0]

        # for simplifying cosine distance calculation below
        U_norm = out / out.norm(dim=-1, keepdim=True)

        # extract and normalize feats
        Y = feats.squeeze()
        Y_norm = Y / Y.norm(dim=-1, keepdim=True)

        # # positive and negative sampling
        # ps_dists = out.mul(feats_).sum(dim=-1, keepdim=True)
        # ns_dists = out.matmul(feats_.t())

        # Implementation from original paper (Max-Margin)
        errors = U_norm.matmul(Y_norm.t())
        diag = errors.diag()
        # all contrastive images for each sentence
        loss_s = self.margin - errors + diag.unsqueeze(-1)
        loss_s = torch.max(loss_s, torch.zeros_like(loss_s))
        # all contrastive sentences for each image
        loss_i = self.margin - errors + diag.unsqueeze(0)
        loss_i = torch.max(loss_i, torch.zeros_like(loss_i))
        # total loss
        loss_tot = loss_s + loss_i
        loss_tot[range(B), range(B)] = 0.0

        # one for positive sample
        return {'loss': loss_tot.mean()}
