# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple, Union

import torch
import torch.nn as nn
from mmcls.evaluation.metrics import Accuracy
from mmcls.models import ClsHead
from mmcls.structures import ClsDataSample
from mmcv.cnn import build_norm_layer
from mmengine.structures import LabelData

from mmselfsup.registry import MODELS
from ..utils import MultiPooling


@MODELS.register_module()
class MultiClsHead(ClsHead):
    """Multiple classifier heads.

    This head inputs feature maps from different stages of backbone, average
    pools each feature map to around 9000 dimensions, and then appends a
    linear classifier at each stage to predict corresponding class scores.

    Args:
        backbone (str): Specify which backbone to use, only support
            ResNet50. Defaults to 'resnet50'.
        in_indices (Sequence[int]): Input from which stages.
            Defaults to (0, 1, 2, 3, 4).
        pool_type (str): 'adaptive' or 'specified'. If set to
            'adaptive', use adaptive average pooling, otherwise use specified
            pooling params. Defaults to 'adaptive'.
        num_classes (int): Number of classes. Defaults to 1000.
        loss (dict): The dict of loss information. Defaults to
            'mmcls.models.CrossEntro): Whether to unpool the features
            from last layer. Defaults to False.
        cal_acc (bool): Whether to calculate accuracy during training.
            If you use batch augmentations like Mixup and CutMix during
            training, it is pointless to calculate accuracy.
            Defaults to False.
        topk (int | Tuple[int]): Top-k accuracy. Defaults to ``(1, )``.
        norm_cfg (dict): Dict to construct and config norm layer.
            Defaults to ``dict(type='BN')``.
        init_cfg (dict or List[dict]): Initialization config dict.
            Defaults to ``[
            dict(type='Normal', std=0.01, layer='Linear'),
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
            ]``
    """

    FEAT_CHANNELS = {'resnet50': [64, 256, 512, 1024, 2048]}
    FEAT_LAST_UNPOOL = {'resnet50': 2048 * 7 * 7}

    def __init__(
        self,
        backbone: str = 'resnet50',
        in_indices: Sequence[int] = (0, 1, 2, 3, 4),
        pool_type: str = 'adaptive',
        num_classes: int = 1000,
        loss: dict = dict(type='mmcls.CrossEntropyLoss', loss_weight=1.0),
        with_last_layer_unpool: bool = False,
        cal_acc: bool = False,
        topk: Union[int, Tuple[int]] = (1, ),
        norm_cfg: dict = dict(type='BN'),
        init_cfg: Union[dict, List[dict]] = [
            dict(type='Normal', std=0.01, layer='Linear'),
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ]
    ) -> None:
        super().__init__(loss=loss, init_cfg=init_cfg)
        assert norm_cfg['type'] in ['BN', 'SyncBN', 'GN', 'null']
        self.in_indices = in_indices
        self.with_norm = norm_cfg['type'] != 'null'

        # add fc if with_last_layer_unpool
        self.with_last_layer_unpool = with_last_layer_unpool
        if with_last_layer_unpool:
            self.fcs.append(
                nn.Linear(self.FEAT_LAST_UNPOOL[backbone], num_classes))

        self.multi_pooling = MultiPooling(pool_type, in_indices, backbone)

        # build norm and fc layers
        if self.with_norm:
            self.norms = nn.ModuleList([
                build_norm_layer(norm_cfg, self.FEAT_CHANNELS[backbone][i])[1]
                for i in in_indices
            ])
        self.fcs = nn.ModuleList([
            nn.Linear(self.multi_pooling.POOL_DIMS[backbone][i], num_classes)
            for i in in_indices
        ])

        self.cal_acc = cal_acc
        self.topk = topk
        # build loss
        self.loss_module = MODELS.build(loss)

    def forward(self, feats: Union[list, tuple]) -> list:
        """Compute multi-head scores.

        Args:
            feats (Sequence[torch.Tensor]): Feature maps of backbone,
                each tensor has shape (N, C, H, W).

        Returns:
            List[torch.Tensor]: A list of class scores.
        """
        assert isinstance(feats, (list, tuple))
        if self.with_last_layer_unpool:
            last_feats = feats[-1]

        feats = self.multi_pooling(feats)

        if self.with_norm:
            feats = [norm(x) for norm, x in zip(self.norms, feats)]

        if self.with_last_layer_unpool:
            feats.append(last_feats)

        feats = [x.view(x.size(0), -1) for x in feats]
        cls_score = [fc(x) for fc, x in zip(self.fcs, feats)]
        return cls_score

    def loss(self, feats: Sequence[torch.Tensor],
             data_samples: List[ClsDataSample], **kwargs) -> dict:
        """Calculate losses from the extracted features.

        Args:
            x (Sequence[torch.Tensor]): Feature maps of backbone,
                each tensor has shape (N, C, H, W).
            gt_label (torch.Tensor): The ground truth label.

        Returns:
            Dict[str, torch.Tensor]: Dict of loss and accuracy.
        """
        cls_score = self(feats)

        # Unpack data samples and pack targets
        if 'score' in data_samples[0].gt_label:
            # Batch augmentation may convert labels to one-hot format scores.
            target = torch.stack(
                [data_sample.gt_label.score for data_sample in data_samples])
        else:
            target = torch.cat(
                [data_sample.gt_label.label for data_sample in data_samples])

        # compute loss and accuracy
        losses = dict()
        for i, score in zip(self.in_indices, cls_score):
            losses[f'loss.{i + 1}'] = self.loss_module(score, target)
            if self.cal_acc:
                acc = Accuracy.calculate(score, target, topk=self.topk)
                losses.update({
                    f'accuracy.{i+1}.top-{k}': a
                    for k, a in zip(self.topk, acc)
                })

        return losses

    def predict(self, feats: Sequence[torch.Tensor],
                data_samples: List[ClsDataSample]) -> List[ClsDataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The extracted features.
            data_samples (List[BaseDataElement], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples.

        Returns:
            List[BaseDataElement]: The data samples containing annotation,
                prediction, etc.
        """
        # multi-head scores
        pred_scores = self(feats)

        assert data_samples is not None
        for i, head_scores in zip(self.in_indices, pred_scores):
            for data_sample, score in zip(data_samples, head_scores.detach()):
                tmp_pred_score = LabelData(score=score)
                name = f'head{i}_pred_label'
                data_sample.set_field(tmp_pred_score, name, dtype=LabelData)

        return data_samples
