from torch import nn

class MaxMarginLoss(nn.Module):
    def __init__(self, margin=0.5, constrastive_type='intruder'):
        super().__init__()

        self.constrastive_type = constrastive_type.lower()

        assert margin > 0., "margin must be positive"
        assert self.constrastive_type in ('all', 'intruder'), \
            f"Unknown constrastive_type '{self.constrastive_type}'"

        self.margin = margin
        self.forward = getattr(self, f'_forward_{self.constrastive_type}')

    def _forward_all(self, O, Y):
        B = O.shape[0]
        corrects = O[range(B), Y].unsqueeze(-1)    # Bx1
        
        # calculate loss for all canditees over vocabulary
        loss = self.margin + O - corrects   # BxV
        # mask corrects
        loss[range(B), Y] = 0.0
        # mean loss except correct one
        loss = loss.sum(dim=-1) / (loss.shape[-1] - 1)

        # suppress loss　negative values and padding
        loss[loss < 0] = 0.0
        loss[Y == 0] = 0.0

        return loss.sum()

    def _forward_intruder(self, O, Y):
        B = O.shape[0]
        corrects = O[range(B), Y].unsqueeze(-1)    # Bx1
        
        # Lazaridou, Dinu, Baroni - ACL 2015 - Hubness and Pollution Delving into Cross-Space Mapping for Zero-Shot Learning
        # max() returns both exact values AND indicators, only first one is needed
        loss = self.margin + (O - corrects).max(dim=-1)[0]

        # suppress loss　negative values and padding
        loss[loss < 0] = 0.0
        loss[Y == 0] = 0.0

        return loss.sum()

    def __repr__(self):
        return self.__class__.__name__ + \
            f"(margin={self.margin}, constrastive_type={self.constrastive_type})" 