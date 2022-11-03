# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
from mmengine.structures import BaseDataElement

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel


@MODELS.register_module()
class MAE(BaseModel):
    """MAE.

    Implementation of `Masked Autoencoders Are Scalable Vision Learners
    <https://arxiv.org/abs/2111.06377>`_.
    """

    def extract_feat(self, inputs: List[torch.Tensor],
                     **kwarg) -> Tuple[torch.Tensor]:
        """The forward function to extract features from neck.

        Args:
            inputs (List[torch.Tensor]): The input images.

        Returns:
            Tuple[torch.Tensor]: Neck outputs.
        """
        latent, _, ids_restore = self.backbone(inputs[0])
        pred = self.neck(latent, ids_restore)
        return pred

    def predict(self,
                inputs: List[torch.Tensor],
                data_samples: Optional[List[SelfSupDataSample]] = None,
                **kwargs) -> SelfSupDataSample:
        """The forward function in testing. It is mainly for image
        reconstruction.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            SelfSupDataSample: The prediction from model.
        """

        latent, mask, ids_restore = self.backbone(inputs[0])
        pred = self.neck(latent, ids_restore)

        pred = self.head.unpatchify(pred)
        pred = torch.einsum('nchw->nhwc', pred).detach().cpu()

        mask = mask.detach()
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
