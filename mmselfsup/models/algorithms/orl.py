# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmengine.logging import print_log

from mmselfsup.registry import MODELS
from .. import builder
from .base import BaseModel


@MODELS.register_module()
class SelectiveSearch(nn.Module):
    """Selective-search proposal generation."""

    def __init__(self, **kwargs):
        super(SelectiveSearch, self).__init__()

    def forward_test(self, bbox, **kwargs):
        assert bbox.dim() == 3, \
            'Input bbox must have 3 dims, got: {}'.format(bbox.dim())
        # bbox: 1xBx4
        return dict(bbox=bbox.cpu())

    def forward(self, mode='test', **kwargs):
        assert mode == 'test', \
            'Support test inference mode only, got: {}'.format(mode)
        return self.forward_test(**kwargs)


@MODELS.register_module()
class ORL(BaseModel):
    """ORL.

    Args:
        backbone (dict): Config dict for module of
            backbone ConvNet.
        neck (dict):
            Config dict for module of deep features to compact feature vectors.
            Default: None.
        head (dict):
            Config dict for module of loss functions.
            Default: None.
        pretrained (str, optional):
            Path to pre-trained weights.
            Default: None.
        base_momentum (float):
            The base momentum coefficient for the target network.
            Default: 0.99.
        global_loss_weight (float):
            Loss weight for global image branch. Default: 1.
        local_intra_loss_weight (float):
            Loss weight for local intra-roi branch. Default: 1.
        local_inter_loss_weight (float):
            Loss weight for local inter-roi branch. Default: 1.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 base_momentum=0.99,
                 global_loss_weight=1.,
                 local_intra_loss_weight=1.,
                 local_inter_loss_weight=1.,
                 **kwargs):
        super(ORL, self).__init__()
        self.global_loss_weight = global_loss_weight
        self.loc_intra_weight = local_intra_loss_weight
        self.loc_inter_weight = local_inter_loss_weight
        self.online_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.target_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))

        self.backbone = self.online_net[0]
        self.neck = self.online_net[1]
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.global_head = builder.build_head(head)
        self.loc_intra_head = builder.build_head(head)
        self.loc_inter_head = builder.build_head(head)
        self.init_weights(pretrained=pretrained)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.online_net[0].init_weights(pretrained=pretrained)  # backbone
        self.online_net[1].init_weights(init_linear='kaiming')  # projection
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)
        # init the predictor in the head
        self.global_head.init_weights(init_linear='kaiming')
        self.loc_intra_head.init_weights(init_linear='kaiming')
        self.loc_inter_head.init_weights(init_linear='kaiming')

    @torch.no_grad()
    def _momentum_update(self):
        """Momentum update of the target network."""
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                             param_ol.data * (1. - self.momentum)

    @torch.no_grad()
    def momentum_update(self):
        self._momentum_update()

    def forward_train(self, img, ipatch, kpatch):
        """Forward computation during training.

        Args:
        img (Tensor):
            Input of two concatenated images with shape (N, 2, C, H, W).
            Typically should be mean centered and std scaled.
        ipatch (Tensor):
            Input of two concatenated intra-RoI patches with shape
            (N, 2, C, H, W). Typically should be mean centered and std scaled.
        kpatch (Tensor):
            Input of two concatenated inter-RoI patches with shape
            (N, 2, C, H, W). Typically should be mean centered and std scaled.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, \
            'Input must have 5 dims, got: {}'.format(img.dim())
        img_v1 = img[:, 0, ...].contiguous()
        img_v2 = img[:, 1, ...].contiguous()
        assert ipatch.dim() == 5, \
            'Input must have 5 dims, got: {}'.format(ipatch.dim())
        ipatch_v1 = ipatch[:, 0, ...].contiguous()
        ipatch_v2 = ipatch[:, 1, ...].contiguous()
        assert kpatch.dim() == 5, \
            'Input must have 5 dims, got: {}'.format(kpatch.dim())
        kpatch_v1 = kpatch[:, 0, ...].contiguous()
        kpatch_v2 = kpatch[:, 1, ...].contiguous()
        # compute online features
        global_online_v1 = self.online_net(img_v1)[0]
        global_online_v2 = self.online_net(img_v2)[0]
        loc_intra_v1 = self.online_net(ipatch_v1)[0]
        loc_intra_v2 = self.online_net(ipatch_v2)[0]
        loc_inter_v1 = self.online_net(kpatch_v1)[0]
        loc_inter_v2 = self.online_net(kpatch_v2)[0]
        # compute target features
        with torch.no_grad():
            global_target_v1 = self.target_net(img_v1)[0].clone().detach()
            global_target_v2 = self.target_net(img_v2)[0].clone().detach()
            loc_intra_tar_v1 = self.target_net(ipatch_v1)[0].clone().detach()
            loc_intra_tar_v2 = self.target_net(ipatch_v2)[0].clone().detach()
            loc_inter_tar_v1 = self.target_net(kpatch_v1)[0].clone().detach()
            loc_inter_tar_v2 = self.target_net(kpatch_v2)[0].clone().detach()
        # compute losses
        global_loss =\
            self.global_head(global_online_v1, global_target_v2)['loss'] + \
            self.global_head(global_online_v2, global_target_v1)['loss']

        local_intra_loss =\
            self.loc_intra_head(loc_intra_v1, loc_intra_tar_v2)['loss'] + \
            self.loc_intra_head(loc_intra_v2, loc_intra_tar_v1)['loss']
        local_inter_loss = \
            self.loc_inter_head(loc_inter_v1, loc_inter_tar_v2)['loss'] + \
            self.loc_inter_head(loc_inter_v2, loc_inter_tar_v1)['loss']
        losses = dict()
        losses['loss_global'] = self.global_loss_weight * global_loss
        losses['loss_local_intra'] = self.loc_intra_weight * local_intra_loss
        losses['loss_local_inter'] = self.loc_inter_weight * local_inter_loss
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
            raise Exception('No such mode: {}'.format(mode))
