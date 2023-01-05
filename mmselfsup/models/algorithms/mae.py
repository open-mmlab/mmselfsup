# Copyright (c) OpenMMLab. All rights reserved.
<<<<<<< HEAD
from typing import Dict, List, Optional, Tuple

import torch
from mmengine.structures import BaseDataElement

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
=======
from typing import Dict, Optional, Tuple

import torch

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
>>>>>>> upstream/master
from .base import BaseModel


@MODELS.register_module()
class MAE(BaseModel):
    """MAE.

    Implementation of `Masked Autoencoders Are Scalable Vision Learners
    <https://arxiv.org/abs/2111.06377>`_.
<<<<<<< HEAD
    """

    def extract_feat(self,
                     inputs: List[torch.Tensor],
                     data_samples: Optional[List[SelfSupDataSample]] = None,
                     **kwarg) -> Tuple[torch.Tensor]:
        """The forward function to extract features from neck.

        Args:
            inputs (List[torch.Tensor]): The input images.

        Returns:
            Tuple[torch.Tensor]: Neck outputs.
        """
        latent, mask, ids_restore = self.backbone(inputs[0])
=======

    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        init_cfg (dict, optional): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 init_cfg: Optional[dict] = None) -> None:
        super().__init__(init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        self.neck.num_patches = self.backbone.num_patches
        assert head is not None
        self.head = build_head(head)

    def init_weights(self):
        super().init_weights()

    def extract_feat(self, img: torch.Tensor) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            img (torch.Tensor): Input images of shape (N, C, H, W).
        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """
        return self.backbone(img)

    def forward_train(self, img: torch.Tensor,
                      **kwargs) -> Dict[str, torch.Tensor]:
        """Forward computation during training.

        Args:
            img (torch.Tensor): Input images of shape (N, C, H, W).
            kwargs: Any keyword arguments to be used to forward.
        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        latent, mask, ids_restore = self.backbone(img)
>>>>>>> upstream/master
        pred = self.neck(latent, ids_restore)
        self.mask = mask
        return pred

    def reconstruct(self,
                    features: torch.Tensor,
                    data_samples: Optional[List[SelfSupDataSample]] = None,
                    **kwargs) -> SelfSupDataSample:
        """The function is for image reconstruction.

        Args:
            features (torch.Tensor): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            SelfSupDataSample: The prediction from model.
        """
        mean = kwargs['mean']
        std = kwargs['std']
        features = features * std + mean

        pred = self.head.unpatchify(features)
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

        mask = self.mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.head.patch_size**2 *
                                         3)  # (N, H*W, p*p*3)
        mask = self.head.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        results = SelfSupDataSample()
        results.mask = BaseDataElement(**dict(value=mask))
        results.pred = BaseDataElement(**dict(value=pred))

        return results

    def loss(self, inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        # ids_restore: the same as that in original repo, which is used
        # to recover the original order of tokens in decoder.
        latent, mask, ids_restore = self.backbone(inputs[0])
        pred = self.neck(latent, ids_restore)
        loss = self.head(pred, inputs[0], mask)
        losses = dict(loss=loss)
        return losses

    def forward_test(self, img: torch.Tensor,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward computation during testing.

        Args:
            img (torch.Tensor): Input images of shape (N, C, H, W).
            kwargs: Any keyword arguments to be used to forward.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output of model test.
                - mask: Mask used to mask image.
                - pred: The output of neck.
        """
        latent, mask, ids_restore = self.backbone(img)
        pred = self.neck(latent, ids_restore)

        pred = self.head.unpatchify(pred)
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, self.head.patch_size**2 *
                                         3)  # (N, H*W, p*p*3)
        mask = self.head.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

        return mask, pred
