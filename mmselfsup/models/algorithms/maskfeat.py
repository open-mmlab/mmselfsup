# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import torch
from mmengine.structures import BaseDataElement

from mmselfsup.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from .base import BaseModel


@MODELS.register_module()
class MaskFeat(BaseModel):
    """MaskFeat.

    Implementation of `Masked Feature Prediction for Self-Supervised Visual
    Pre-Training <https://arxiv.org/abs/2112.09133>`_.
    """

    def extract_feat(self,
                     inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     compute_hog: bool = True,
                     **kwarg) -> Tuple[torch.Tensor]:
        """The forward function to extract features from neck.

        Args:
            inputs (List[torch.Tensor]): The input images and mask.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.
            compute_hog (bool): Whether to compute hog during extraction. If
                True, the batch size of inputs need to be 1. Defaults to True.

        Returns:
            Tuple[torch.Tensor]: Neck outputs.
        """
        img = inputs[0]
        self.mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])
        latent = self.backbone(img, self.mask)
        B, L, C = latent.shape
        pred = self.neck([latent.view(B * L, C)])
        pred = pred[0].view(B, L, -1)

        # compute hog
        if compute_hog:
            assert img.size(0) == 1, 'Currently only support batch size 1.'
            _ = self.target_generator(img)
            hog_image = torch.from_numpy(
                self.target_generator.generate_hog_image(
                    self.target_generator.out)).unsqueeze(0).unsqueeze(0)
            self.target = hog_image.expand(-1, 3, -1, -1)

        return pred[:, 1:, :]  # remove cls token

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
        img = inputs[0]
        mask = torch.stack(
            [data_sample.mask.value for data_sample in data_samples])
        mask = mask.to(torch.bool)

        latent = self.backbone(img, mask)
        B, L, C = latent.shape
        pred = self.neck([latent.view(B * L, C)])
        pred = pred[0].view(B, L, -1)
        hog = self.target_generator(img)

        loss = self.head(pred, hog, mask)
        losses = dict(loss=loss)
        return losses

    def reconstruct(self,
                    features: List[torch.Tensor],
                    data_samples: Optional[List[SelfSupDataSample]] = None,
                    **kwargs) -> SelfSupDataSample:
        """The function is for image reconstruction.

        Args:
            features (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            SelfSupDataSample: The prediction from model.
        """

        # recover to HOG description from feature embeddings
        unfold_size = self.target_generator.unfold_size
        tmp4 = features.unflatten(2,
                                  (features.shape[2] // unfold_size**2,
                                   unfold_size, unfold_size))  # 1,196,27,2,2
        tmp3 = tmp4.unflatten(1, self.backbone.patch_resolution)

        b, p1, p2, c_nbins, _, _ = tmp3.shape  # 1,14,14,27,2,2
        tmp2 = tmp3.permute(0, 1, 2, 5, 3, 4).reshape(
            (b, p1, p2 * unfold_size, c_nbins, unfold_size))
        tmp1 = tmp2.permute(0, 1, 4, 2, 3).reshape(
            (b, p1 * unfold_size, p2 * unfold_size, c_nbins))
        tmp0 = tmp1.permute(0, 3, 1, 2)  # 1,27,28,28
        hog_out = tmp0.unflatten(1,
                                 (int(c_nbins // self.target_generator.nbins),
                                  self.target_generator.nbins))  # 1,3,9,28,28

        # generate predction of HOG
        hog_image = torch.from_numpy(
            self.target_generator.generate_hog_image(hog_out))
        hog_image = hog_image.unsqueeze(0).unsqueeze(0)
        pred = torch.einsum('nchw->nhwc', hog_image).expand(-1, -1, -1,
                                                            3).detach().cpu()

        # transform patch mask to pixel mask
        mask = self.mask
        patch_dim_1 = int(self.backbone.patch_embed.init_input_size[0] //
                          self.backbone.patch_resolution[0])
        patch_dim_2 = int(self.backbone.patch_embed.init_input_size[1] //
                          self.backbone.patch_resolution[1])
        mask = mask.repeat_interleave(
            patch_dim_1, dim=1).repeat_interleave(
                patch_dim_2, dim=2).unsqueeze(-1).repeat(1, 1, 1, 3)
        # 1 is removing, 0 is keeping
        mask = mask.detach().cpu()

        results = SelfSupDataSample()
        results.mask = BaseDataElement(**dict(value=mask))
        results.pred = BaseDataElement(**dict(value=pred))

        return results
