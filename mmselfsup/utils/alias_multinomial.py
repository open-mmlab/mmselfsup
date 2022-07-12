# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn


class AliasMethod(nn.Module):
    """The alias method for sampling.

    From: https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

    Args:
        probs (torch.Tensor): Sampling probabilities.
    """  # noqa: E501

    def __init__(self, probs: torch.Tensor) -> None:
        super().__init__()
        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.register_buffer('prob', torch.zeros(K))
        self.register_buffer('alias', torch.LongTensor([0] * K))

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K * prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller + larger:
            self.prob[last_one] = 1

    def draw(self, N: int) -> None:
        """Draw N samples from multinomial.

        Args:
            N (int): Number of samples.

        Returns:
            torch.Tensor: Samples.
        """
        assert N > 0
        K = self.alias.size(0)
        kk = torch.zeros(
            N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1 - b).long())

        return oq + oj
