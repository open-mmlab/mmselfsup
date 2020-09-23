import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class RotationPred(nn.Module):
    """Rotation prediction.

    Implementation of "Unsupervised Representation Learning
    by Predicting Image Rotations (https://arxiv.org/abs/1803.07728)".

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        head (dict): Config dict for module of loss functions. Default: None.
        pretrained (str, optional): Path to pre-trained weights. Default: None.
    """

    def __init__(self, backbone, head=None, pretrained=None):
        super(RotationPred, self).__init__()
        self.backbone = builder.build_backbone(backbone)
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
        self.head.init_weights(init_linear='kaiming')

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

    def forward_train(self, img, rot_label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            rot_label (Tensor): Labels for the rotations.
            kwargs: Any keyword arguments to be used to forward.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.forward_backbone(img)
        outs = self.head(x)
        loss_inputs = (outs, rot_label)
        losses = self.head.loss(*loss_inputs)
        return losses

    def forward_test(self, img, **kwargs):
        x = self.forward_backbone(img)  # tuple
        outs = self.head(x)
        keys = ['head{}'.format(i) for i in range(len(outs))]
        out_tensors = [out.cpu() for out in outs]  # NxC
        return dict(zip(keys, out_tensors))

    def forward(self, img, rot_label=None, mode='train', **kwargs):
        if mode != "extract" and img.dim() == 5:  # Nx4xCxHxW
            assert rot_label.dim() == 2  # Nx4
            img = img.view(
                img.size(0) * img.size(1), img.size(2), img.size(3),
                img.size(4))  # (4N)xCxHxW
            rot_label = torch.flatten(rot_label)  # (4N)
        if mode == 'train':
            return self.forward_train(img, rot_label, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
