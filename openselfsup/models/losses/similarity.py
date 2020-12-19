import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module
class NegativeCosineSimilarityLoss(nn.Module):
    """Head for negative cosine similarity loss.

    Args:
        stop_gradient (bool): Whether to stop gradient of the right
            branch. Default: Ture.
        version (enumerate): original or simplified. The latter is
            faster.
    """

    def __init__(self, stop_gradient=True, version='original'):
        super().__init__()
        self.stop_gradient = stop_gradient
        if version not in ('original', 'simplified'):
            raise ValueError("Unkown version specified")
        self.version = version

    def forward(self, p, z):
        if self.version == 'original':
            if self.stop_gradient:
                z = z.detach()  # stop gradient
            p = F.normalize(p, dim=1)  # l2-normalize
            z = F.normalize(z, dim=1)  # l2-normalize
            return -(p * z).sum(dim=1).mean()

        elif self.version == 'simplified':  # same thing, much faster.
            if self.stop_gradient:
                return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
            else:
                return - F.cosine_similarity(p, z, dim=-1).mean()
        else:
            raise Exception
