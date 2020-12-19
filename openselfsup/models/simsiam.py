import torch
import torch.nn as nn
import torch.nn.functional as F

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class SimSiam(nn.Module):
    """SimSiam.

    Implementation of "Exploring Simple Siamese Representation Learning
     (http://arxiv.org/abs/2011.10566)".
    Part of the code is borrowed from:
    "https://github.com/PatrickHua/SimSiam/blob/main/models/simsiam.py".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for Projection MLP.
            Default: None.
        head (dict): Config dict for Prediction MLP and loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        self.backbone: nn.Module = builder.build_backbone(backbone)
        self.neck: nn.Module = builder.build_neck(neck)
        self.head: nn.Module = builder.build_head(head)
        self.init_weights(pretrained=pretrained)
        self.encoder = nn.Sequential(
            self.backbone,
            self.neck
        )

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            "Input must have 5 dims, got: {}".format(img.dim())
        im_1 = img[:, 0, ...].contiguous()
        im_2 = img[:, 1, ...].contiguous()

        f, h = self.encoder, self.head
        z1, z2 = f(im_1), f(im_2)
        losses = self.head(z1, z2)
        return losses

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
