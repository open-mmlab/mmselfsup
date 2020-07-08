import torch.nn as nn

from ..utils import accuracy
from ..registry import HEADS


@HEADS.register_module
class NonLinearClsHead(nn.Module):
    """Non-linear classifier head, with two fc layers
    (same as Alexnet fc7-fc8).
    """

    def __init__(self,
                 with_avg_pool=False,
                 in_channels=2048,
                 hid_channels=4096,
                 num_classes=1000):
        super(NonLinearClsHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls = nn.Sequential(
            nn.Linear(self.in_channels * 2, self.hid_channels),
            nn.BatchNorm1d(self.hid_channels, momentum=0.003),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(self.hid_channels, self.num_classes))

    def init_weights(self):
        self.fc_cls[0].weight.data.normal_(0, 0.005)
        self.fc_cls[0].bias.data.fill_(0.1)
        self.fc_cls[1].weight.data.fill_(1)
        self.fc_cls[1].bias.data.zero_()
        self.fc_cls[4].weight.data.normal_(0, 0.005)
        self.fc_cls[4].bias.data.zero_()

    def forward(self, x):
        if self.with_avg_pool:
            assert x.dim() == 4, \
                "Tensor must has 4 dims, got: {}".format(x.dim())
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        return [cls_score]

    def loss(self, cls_score, labels):
        losses = dict()
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        losses['loss'] = self.criterion(cls_score[0], labels)
        losses['acc'] = accuracy(cls_score[0], labels)
        return losses
