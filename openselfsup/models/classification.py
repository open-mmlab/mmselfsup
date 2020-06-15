import numpy as np

import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS
from .utils import Sobel


@MODELS.register_module
class Classification(nn.Module):

    def __init__(self,
                 backbone,
                 frozen_backbone=False,
                 with_sobel=False,
                 head=None,
                 pretrained=None):
        super(Classification, self).__init__()
        self.with_sobel = with_sobel
        if with_sobel:
            self.sobel_layer = Sobel()
        self.backbone = builder.build_backbone(backbone)
        if frozen_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        if head is not None:
            self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.head.init_weights()

    def forward_backbone(self, img):
        """Forward backbone

        Returns:
            x (tuple): backbone outputs
        """
        if self.with_sobel:
            img = self.sobel_layer(img)
        x = self.backbone(img)
        return x

    def forward_train(self, img, gt_label, **kwargs):
        x = self.forward_backbone(img)
        outs = self.head(x)
        loss_inputs = (outs, gt_label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, img, **kwargs):
        x = self.forward_backbone(img)  # tuple
        outs = self.head(x)
        keys = ['head{}'.format(i) for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def aug_test(self, imgs):
        raise NotImplemented
        outs = np.mean([self.head(x) for x in self.forward_backbone(imgs)],
                       axis=0)
        return outs

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
