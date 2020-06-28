import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from .. import builder

@HEADS.register_module
class LatentPredictHead(nn.Module):
    '''Head for contrastive learning.
    '''

    def __init__(self, predictor):
        super(LatentPredictHead, self).__init__()
        self.predictor = builder.build_neck(predictor)

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, input, target):
        '''
        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.
        '''
        N = input.size(0)
        pred = self.predictor([input])[0]
        pred_norm = nn.functional.normalize(pred, dim=1)
        target_norm = nn.functional.normalize(target, dim=1)
        loss = 2 - 2 * (pred_norm * target_norm).sum() / N
        return dict(loss=loss)


@HEADS.register_module
class LatentClsHead(nn.Module):
    '''Head for contrastive learning.
    '''

    def __init__(self, predictor):
        super(LatentClsHead, self).__init__()
        self.predictor = nn.Linear(predictor.in_channels,
                                   predictor.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self, init_linear='normal'):
        normal_init(self.predictor, std=0.01)

    def forward(self, input, target):
        '''
        Args:
            input (Tensor): NxC input features.
            target (Tensor): NxC target features.
        '''
        pred = self.predictor(input)
        with torch.no_grad():
            label = torch.argmax(self.predictor(target), dim=1).detach()
        loss = self.criterion(pred, label)
        return dict(loss=loss)
