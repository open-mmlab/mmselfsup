import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .. import builder


@HEADS.register_module
class LatentPredictHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, predictor, size_average=True):
        super(LatentPredictHead, self).__init__()
        self.predictor = builder.build_neck(predictor)
        self.size_average = size_average

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = -2 * (pred_norm * target_norm).sum()
        if self.size_average:
            loss /= input.size(0)
        return dict(loss=loss)


@HEADS.register_module
class LatentClsHead(nn.Module):
    """Head for contrastive learning.
    """

    def __init__(self, predictor):
        super(LatentClsHead, self).__init__()
        self.predictor = nn.Linear(predictor.in_channels,
                                   predictor.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self, init_linear='normal'):
        normal_init(self.predictor, std=0.01)

    def forward(self, input, target):
        """Forward head.

        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        pred = self.predictor(input)
        with torch.no_grad():
            label = torch.argmax(self.predictor(target), dim=1).detach()
        loss = self.criterion(pred, label)
        return dict(loss=loss)


@HEADS.register_module
class SimSiamPredictHead(nn.Module):
    def __init__(self, predictor, loss):
        super().__init__()
        self.predictor = builder.build_neck(predictor)
        self.loss = builder.build_loss(loss)

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, z1, z2):
        """Forward head.

        Args:
            z1 (Tensor): NxC input features.
            z2 (Tensor): NxC target features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        p1, p2 = self.predictor(z1)[0], self.predictor(z2)[0]
        loss = self.loss(p1, z2[0]) / 2 + self.loss(p2, z1[0]) / 2
        return dict(loss=loss)
