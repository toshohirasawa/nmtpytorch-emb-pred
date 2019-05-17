from torch import nn
import torch

class EmbeddingOutput(nn.Module):

    def __init__(self, n_vocab, emb_size, weight=None):
        super().__init__()
        self.n_vocab, self.emb_size = n_vocab, emb_size

        if weight == None:
            weight = torch.randn(n_vocab, emb_size)

        # travarse matrix for easy matmul calclulation in forward
        self.weight = nn.Parameter(weight)
        self.weight.requires_grad = False

    def forward(self, input):
        # grad in this module makes learning unstable
        return input.matmul(self.weight.data.t())

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(n_vocab={self.n_vocab}, emb_size={self.emb_size})'
