# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcls.models import ImageClassifier
from mmcv.runner import auto_fp16

from ..builder import ALGORITHMS


@ALGORITHMS.register_module()
class MMClsImageClassifierWrapper(ImageClassifier):
    """Workaround to use models from mmclassificaiton.

    Since the output of classifier from mmclassification is not compatible
    with mmselfsup's evaluation function. We rewrite some key components
    from mmclassification.

    Args:
         backbone (dict): Config dict for module of backbone.
         neck (dict, optional): Config dict for module of neck.
             Defaults to None.
         head (dict, optional): Config dict for module of loss functions.
             Defaults to None.
         pretrained (str, optional): The path of pre-trained checkpoint.
             Defaults to None.
         train_cfg (dict, optional): Config dict for pre-processing utils,
             e.g. mixup. Defaults to None.
         init_cfg (dict, optional): Config dict for initialization. Defaults
             to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict = None,
                 head: dict = None,
                 pretrained: str = None,
                 train_cfg: dict = None,
                 init_cfg: dict = None):
        super(MMClsImageClassifierWrapper, self).__init__(
            backbone=backbone,
            neck=neck,
            head=head,
            pretrained=pretrained,
            train_cfg=train_cfg,
            init_cfg=init_cfg)

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, mode='train', **kwargs):
        """Forward function of model.

        Calls either forward_train, forward_test or extract_feat function
        according to the mode.
        """
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.extract_feat(img)
        else:
            raise Exception(f'No such mode: {mode}')

    def forward_train(self, img, label, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            label (Tensor): It should be of shape (N, 1) encoding the
                ground-truth label of input images for single label task. It
                shoulf be of shape (N, C) encoding the ground-truth label
                of input images for multi-labels task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if self.augments is not None:
            img, label = self.augments(img, label)

        x = self.extract_feat(img)

        losses = dict()
        loss = self.head.forward_train(x, label)

        losses.update(loss)

        return losses

    def forward_test(self, imgs, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
        """
        kwargs.pop('label', None)
        kwargs.pop('idx', None)
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]
        for var, name in [(imgs, 'imgs')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        if len(imgs) == 1:
            outs = self.simple_test(imgs[0], post_process=False, **kwargs)
            outs = outs if isinstance(outs, list) else [outs]
            keys = [f'head{i}' for i in self.backbone.out_indices]
            out_tensors = [out.cpu() for out in outs]
            return dict(zip(keys, out_tensors))
        else:
            raise NotImplementedError('aug_test has not been implemented')
