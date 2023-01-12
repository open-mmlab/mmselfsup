# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from ..utils import CosineEMA
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
class Correspondence(BaseModel):
    """Correspondence discovery in Stage 2 of ORL.

    Args:
        backbone (dict): Config dict for module of backbone ConvNet.
        neck (dict): Config dict for module of deep features to
            compact feature vectors. Default: None.
        head (dict): Config dict for module of loss functions.
            Default: None.
        pretrained (str, optional): Path to pre-trained weights.
            Default: None.
        base_momentum (float): The base momentum coefficient for
            the target network. Default: 0.99.
        knn_image_num (int): The number of KNN images. Default: 10.
        topk_bbox_ratio (float): The ratio of retrieved top-ranked RoI pairs.
            Default: 0.1.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 base_momentum: float = 0.99,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None,
                 knn_image_num: int = 10,
                 topk_bbox_ratio: float = 0.1) -> None:
        # super(Correspondence, self).__init__()
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            # data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        # create momentum model
        self.online_net = nn.Sequential(self.backbone, self.neck)
        self.target_net = CosineEMA(self.online_net, momentum=base_momentum)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

        self.knn_image_num = knn_image_num
        self.topk_bbox_ratio = topk_bbox_ratio

    def predict(self, img: List[torch.Tensor], bbox: List[torch.Tensor],
                img_keys: dict, bbox_keys: dict, **kwargs) -> dict:

        knn_imgs = [
            img_keys.get('{}nn_img'.format(k))
            for k in range(self.knn_image_num)
        ]
        knn_bboxes = [
            bbox_keys.get('{}nn_bbox'.format(k))
            for k in range(self.knn_image_num)
        ]
        assert img.size(0) == 1, \
            f'batch size must be 1, got: {img.size(0)}'
        assert img.dim() == 5, \
            f'img must have 5 dims, got: {img.dim()}'
        assert bbox.dim() == 3, \
            f'bbox must have 3 dims, got: {bbox.dim()}'
        assert knn_imgs[0].dim() == 5, \
            f'knn_img must have 5 dims, got: {knn_imgs[0].dim()}'
        assert knn_bboxes[0].dim() == 3, \
            f'knn_bbox must have 3 dims, got: {knn_bboxes[0].dim()}'
        img = img.view(
            img.size(0) * img.size(1), img.size(2), img.size(3),
            img.size(4))  # (1B)xCxHxW
        knn_imgs = [
            knn_img.view(
                knn_img.size(0) * knn_img.size(1),
                knn_img.size(2),
                # K (1B)xCxHxW
                knn_img.size(3),
                knn_img.size(4)) for knn_img in knn_imgs
        ]
        with torch.no_grad():
            feat = self.backbone(img)[0].clone().detach()
            knn_feats = [
                self.backbone(knn_img)[0].clone().detach()
                for knn_img in knn_imgs
            ]
            feat = F.adaptive_avg_pool2d(feat, (1, 1))
            knn_feats = [
                F.adaptive_avg_pool2d(knn_feat, (1, 1))
                for knn_feat in knn_feats
            ]
            feat = feat.view(feat.size(0), -1)  # (1B)xC
            knn_feats = [
                knn_feat.view(knn_feat.size(0), -1) for knn_feat in knn_feats
            ]  # K (1B)xC
            feat_norm = F.normalize(feat, dim=1)
            knn_feats_norm = [
                F.normalize(knn_feat, dim=1) for knn_feat in knn_feats
            ]
            # smaps: a list containing K similarity matrix (BxB Tensor)
            smaps = [
                torch.mm(feat_norm, knn_feat_norm.transpose(0, 1))
                for knn_feat_norm in knn_feats_norm
            ]  # K BxB
            top_query_inds = []
            top_key_inds = []
            for smap in smaps:
                topk_num = int(self.topk_bbox_ratio * smap.size(0))
                _, top_ind = torch.topk(smap.flatten(),
                                        topk_num if topk_num > 0 else 1)
                top_query_ind = top_ind // smap.size(1)
                top_key_ind = top_ind % smap.size(1)
                top_query_inds.append(top_query_ind)
                top_key_inds.append(top_key_ind)
            bbox = bbox.view(bbox.size(0) * bbox.size(1),
                             bbox.size(2))  # (1B)x4
            knn_bboxes = [
                knn_bbox.view(
                    knn_bbox.size(0) * knn_bbox.size(1), knn_bbox.size(2))
                for knn_bbox in knn_bboxes
            ]  # K (1B)x4
            # K (topk_bbox_num)x8
            topk_box_pairs_list = [
                torch.cat((bbox[qind], kbox[kind]),
                          dim=1).cpu() for kbox, qind, kind in zip(
                              knn_bboxes, top_query_inds, top_key_inds)
            ]
            knn_bbox_keys = [
                '{}nn_bbox'.format(k) for k in range(len(topk_box_pairs_list))
            ]
            dict1 = dict(intra_bbox=bbox.cpu())
            dict2 = dict(zip(knn_bbox_keys, topk_box_pairs_list))
        # intra_bbox: Bx4, inter_bbox: K (topk_bbox_num)x8
        # B is the number of filtered bboxes, K is the number of knn images,
        return {**dict1, **dict2}

    def forward(self, img, bbox, img_keys, bbox_keys, mode='test', **kwargs):
        if mode == 'test':
            return self.predict(img, bbox, img_keys, bbox_keys, **kwargs)
        else:
            raise Exception('No such mode: {}'.format(mode))


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
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 base_momentum: float = 0.99,
                 pretrained: Optional[str] = None,
                 init_cfg: Optional[Union[List[dict], dict]] = None,
                 global_loss_weight: float = 1.,
                 loc_intra_loss_weight: float = 1.,
                 loc_inter_loss_weight: float = 1.,
                 **kwargs):
        super().__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            init_cfg=init_cfg)
        self.global_loss_weight = global_loss_weight
        self.loc_intra_weight = loc_intra_loss_weight
        self.loc_inter_weight = loc_inter_loss_weight
        self.online_net = nn.Sequential(self.backbone, self.neck)
        self.target_net = CosineEMA(self.online_net, momentum=base_momentum)

        self.loc_intra_head = MODELS.build(head)
        self.loc_inter_head = MODELS.build(head)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

    def loss(self, inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample], **kwargs) -> dict:
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
        assert isinstance(inputs, list)
        global_img, ipatch, kpatch = inputs
        assert global_img.dim() == 5, \
            'Input must have 5 dims, got: {}'.format(global_img.dim())
        img_v1 = global_img[:, 0, ...].contiguous()
        img_v2 = global_img[:, 1, ...].contiguous()
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
            self.head(global_online_v1, global_target_v2) + \
            self.head(global_online_v2, global_target_v1)

        local_intra_loss =\
            self.loc_intra_head(loc_intra_v1, loc_intra_tar_v2) + \
            self.loc_intra_head(loc_intra_v2, loc_intra_tar_v1)
        local_inter_loss = \
            self.loc_inter_head(loc_inter_v1, loc_inter_tar_v2) + \
            self.loc_inter_head(loc_inter_v2, loc_inter_tar_v1)
        losses = dict()
        loss_global = self.global_loss_weight * global_loss
        loss_local_intra = self.loc_intra_weight * local_intra_loss
        loss_local_inter = self.loc_inter_weight * local_inter_loss

        losses = dict(loss=loss_global + loss_local_intra + loss_local_inter)
        return losses

    def forward(self,
                inputs: List[torch.Tensor],
                data_samples: Optional[List[SelfSupDataSample]] = None,
                mode: str = 'tensor'):
        if mode == 'loss':
            return self.loss(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')
