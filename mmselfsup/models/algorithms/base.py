# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16

from mmselfsup.core import SelfSupDataSample
from mmselfsup.utils import get_module_device


class BaseModel(BaseModule, metaclass=ABCMeta):
    """Base model class for self-supervised learning.

    Args:
        preprocess_cfg (Dict): Config to preprocess images.
        init_cfg (Dict, optional): Config to initialize models.
            Defaults to None.
    """

    def __init__(self,
                 preprocess_cfg: Dict,
                 init_cfg: Optional[Dict] = None) -> None:
        super(BaseModel, self).__init__(init_cfg)
        self.fp16_enabled = False
        assert 'mean' in preprocess_cfg
        self.register_buffer(
            'mean_norm',
            torch.tensor(preprocess_cfg.pop('mean')).view(3, 1, 1))
        assert 'std' in preprocess_cfg
        self.register_buffer(
            'std_norm',
            torch.tensor(preprocess_cfg.pop('std')).view(3, 1, 1))
        assert 'to_rgb' in preprocess_cfg
        self.to_rgb = preprocess_cfg.pop('to_rgb')

    @property
    def with_neck(self) -> bool:
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_head(self) -> bool:
        return hasattr(self, 'head') and self.head is not None

    @abstractmethod
    def extract_feat(self, inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     **kwargs) -> object:
        """The forward function to extract features.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.
        """
        raise NotImplementedError('``extract_feat`` should be implemented')

    @abstractmethod
    def forward_train(self, inputs: List[torch.Tensor],
                      data_samples: List[SelfSupDataSample],
                      **kwargs) -> object:
        """The forward function in training
        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.
        """
        raise NotImplementedError('``forward_train`` should be implemented')

    def forward_test(self, inputs: List[torch.Tensor],
                     data_samples: List[SelfSupDataSample],
                     **kwargs) -> object:
        """The forward function in testing
        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.
        """
        raise NotImplementedError('``forward_test`` should be implemented')

    @auto_fp16(apply_to=('data', ))
    def forward(self,
                data: List[Dict],
                return_loss: bool = False,
                extract: bool = False,
                **kwargs) -> object:
        """Forward function of model.

        Calls either forward_train, forward_test or extract_feat function
        according to the mode.

        Args:
            data (List[Dict]): The input data for model.
            return_loss (bool): Train mode or test mode. Defaults to False.
            extract (bool): Whether or not only extract features from model.
                If set to True, the ``return_loss`` will be ignored. Defaults
                to False.
        """
        # preprocess images
        inputs, data_samples = self.preprocss_data(data)

        # Whether or not extract features. If set to True, the ``return_loss``
        # will be ignored.
        if extract:
            return self.extract_feat(
                inputs=inputs, data_samples=data_samples, **kwargs)

        if return_loss:
            losses = self.forward_train(
                inputs=inputs, data_samples=data_samples, **kwargs)
            loss, log_vars = self._parse_losses(losses)
            outputs = dict(loss=loss, log_vars=log_vars)
            return outputs
        else:
            # should be a list of SelfSupDataSample
            return self.forward_test(
                inputs=inputs, data_samples=data_samples, **kwargs)

    def _parse_losses(self, losses: Dict) -> Tuple[torch.Tensor, Dict]:
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (Dict): Raw output of the network, which usually contain
                losses and other necessary information.
        Returns:
            tuple[torch.Tensor, Dict]: (loss, log_vars), loss is the loss
                tensor which may be a weighted sum of all losses, log_vars
                contains all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            elif isinstance(loss_value, dict):
                for name, value in loss_value.items():
                    log_vars[name] = value
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

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

        # reconstruct images into several batches. For example, SimCLR needs
        # two crops for each image, and this code snippet will convert images
        # into two batches, each containing one crop of an image.
        batch_images = []
        for i in range(len(images[0])):
            cur_batch = [img[i] for img in images]
            batch_images.append(torch.stack(cur_batch))

        return batch_images, data_samples
