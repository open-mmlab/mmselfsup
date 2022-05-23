# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
from mmengine.data import InstanceData

from mmselfsup.core import SelfSupDataSample
from mmselfsup.utils import get_module_device
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class RelativeLoc(BaseModel):
    """Relative patch location.

    Implementation of `Unsupervised Visual Representation Learning
    by Context Prediction <https://arxiv.org/abs/1505.05192>`_.

    Args:
        backbone (Dict, optional): Config dict for module of backbone.
            Defaults to None.
        neck (Dict, optional): Config dict for module of deep features
            to compact feature vectors. Defaults to None.
        head (Dict, optional): Config dict for module of loss functions.
            Defaults to None.
        preprocess_cfg (Dict, optional): Config dict to preprocess images.
            Defaults to None.
        init_cfg (Dict or List[Dict], optional): Config dict for weight
            initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: Optional[Dict] = None,
                 neck: Optional[Dict] = None,
                 head: Optional[Dict] = None,
                 preprocess_cfg: Optional[Dict] = None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None,
                 **kwargs) -> None:
        super().__init__(preprocess_cfg=preprocess_cfg, init_cfg=init_cfg)
        assert backbone is not None
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

    def extract_feat(self, inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     **kwargs) -> Tuple[torch.Tensor]:
        """Function to extract features from backbone.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Tuple[torch.Tensor]: backbone outputs.
        """

        x = self.backbone(inputs[0])
        return x

    def forward_train(self, inputs: List[torch.Tensor],
                      data_samples: List[SelfSupDataSample],
                      **kwargs) -> Dict[str, torch.Tensor]:
        """Forward computation during training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """

        x1 = self.backbone(inputs[0])
        x2 = self.backbone(inputs[1])
        x = (torch.cat((x1[0], x2[0]), dim=1), )
        x = self.neck(x)
        outs = self.head(x)

        patch_label = [
            data_sample.patch_label.value for data_sample in data_samples
        ]
        patch_label = torch.flatten(torch.stack(patch_label, 0))
        loss_inputs = (outs, patch_label)
        loss_dict = self.head.loss(*loss_inputs)
        return loss_dict

    def forward_test(self, inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     **kwargs) -> List[SelfSupDataSample]:
        """The forward function in testing.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            List[SelfSupDataSample]: The prediction from model.
        """

        x1 = self.backbone(inputs[0])
        x2 = self.backbone(inputs[1])
        x = (torch.cat((x1[0], x2[0]), dim=1), )
        x = self.neck(x)
        outs = self.head(x)
        keys = [f'head{i}' for i in self.backbone.out_indices]
        outs = [torch.chunk(out, len(outs[0]) // 8, 0) for out in outs]

        for i in range(len(outs[0])):
            prediction_data = {key: out[i] for key, out in zip(keys, outs)}
            prediction = InstanceData(**prediction_data)
            data_samples[i].prediction = prediction
        return data_samples

    def preprocss_data(
            self,
            data: List[Dict]) -> Tuple[List[torch.Tensor], SelfSupDataSample]:
        """Process input data during training, testing or extracting.

        Args:
            data (List[Dict]): The data to be processed, which
                comes from dataloader.

        Returns:
            tuple:  It should contain 2 item.
            - batch_images (List[torch.Tensor]): The batch image tensor.
            - data_samples (List[SelfSupDataSample], Optional): The Data
            Samples. It usually includes information such as
            `gt_label`. Return None If the input data does not
            contain `data_sample`.
        """

        # data_['inputs] is a list
        images = [data_['inputs'] for data_ in data]
        data_samples = [data_['data_sample'] for data_ in data]

        device = get_module_device(self)
        data_samples = [data_sample.to(device) for data_sample in data_samples]
        images = [[img_.to(device) for img_ in img] for img in images]

        # convert images to rgb
        if self.to_rgb and images[0][0].size(0) == 3:
            images = [[img_[[2, 1, 0], ...] for img_ in img] for img in images]

        # normalize images
        images = [[(img_ - self.mean_norm) / self.std_norm for img_ in img]
                  for img in images]

        # reconstruct images into several batches. RelativeLoc needs
        # nine crops for each image, and this code snippet will convert images
        # into nine batches, each containing one crop of an image.
        batch_images = []
        for i in range(len(images[0])):
            cur_batch = [img[i] for img in images]
            batch_images.append(torch.stack(cur_batch))

        img1 = torch.stack(batch_images[1:], 1)  # Nx8xCxHxW
        img1 = img1.view(
            img1.size(0) * img1.size(1), img1.size(2), img1.size(3),
            img1.size(4))  # (8N)xCxHxW
        img2 = torch.unsqueeze(batch_images[0], 1).repeat(1, 8, 1, 1,
                                                          1)  # Nx8xCxHxW
        img2 = img2.view(
            img2.size(0) * img2.size(1), img2.size(2), img2.size(3),
            img2.size(4))  # (8N)xCxHxW
        batch_images = [img1, img2]

        return batch_images, data_samples
