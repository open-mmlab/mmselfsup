import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class RelativeLoc(nn.Module):
    """Relative patch location.

    Implementation of "Unsupervised Visual Representation Learning
    by Context Prediction (https://arxiv.org/abs/1505.05192)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self, backbone, neck=None, head=None, pretrained=None):
        super(RelativeLoc, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        if head is not None:
            self.head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights(init_linear='normal')
        self.head.init_weights(init_linear='normal', std=0.005)

    def forward_backbone(self, img):
        """Forward backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, patch_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            patch_label (Tensor): Labels for the relative patch locations.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        img1, img2 = torch.chunk(img, 2, dim=1)
        x1 = self.forward_backbone(img1)  # tuple
        x2 = self.forward_backbone(img2)  # tuple
        x = (torch.cat((x1[0], x2[0]), dim=1),)
        x = self.neck(x)
        outs = self.head(x)
        loss_inputs = (outs, patch_label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, img, **kwargs):
        img1, img2 = torch.chunk(img, 2, dim=1)
        x1 = self.forward_backbone(img1)  # tuple
        x2 = self.forward_backbone(img2)  # tuple
        x = (torch.cat((x1[0], x2[0]), dim=1),)
        x = self.neck(x)
        outs = self.head(x)
        keys = ['head{}'.format(i) for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]
        return dict(zip(keys, out_tensors))

    def forward(self, img, patch_label=None, mode='train', **kwargs):
        if mode != "extract" and img.dim() == 5:  # Nx8x(2C)xHxW
            assert patch_label.dim() == 2  # Nx8
            img = img.view(
                img.size(0) * img.size(1), img.size(2), img.size(3),
                img.size(4))  # (8N)x(2C)xHxW
            patch_label = torch.flatten(patch_label)  # (8N)
        if mode == 'train':
            return self.forward_train(img, patch_label, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
