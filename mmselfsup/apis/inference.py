# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcv.utils import build_from_cfg
from PIL import Image
from torch import nn
from torchvision.transforms import Compose

from mmselfsup.datasets import PIPELINES
from mmselfsup.models import build_algorithm


def init_model(config: Union[str, mmcv.Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               options: Optional[dict] = None) -> nn.Module:
    """Initialize an model from config file.

    Args:
        config (str or :obj:``mmcv.Config``): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights. Defaults to None.
        device (str): The device where the model will be put on.
            Default to 'cuda:0'.
        options (dict, optional): Options to override some settings in the used
            config. Defaults to None.
    Returns:
        nn.Module: The initialized model.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if options is not None:
        config.merge_from_dict(options)
    model = build_algorithm(config.model)
    if checkpoint is not None:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_model(
        model: nn.Module,
        data: Image) -> Tuple[torch.Tensor, Union[torch.Tensor, dict]]:
    """Inference an image with the model.
    Args:
        model (nn.Module): The loaded model.
        data (PIL.Image): The loaded image.
    Returns:
        Tuple[torch.Tensor, Union(torch.Tensor, dict)]: Output of model
            inference.
            - data (torch.Tensor): The loaded image to input model.
            - output (torch.Tensor, dict[str, torch.Tensor]): the output
                of test model.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [
        build_from_cfg(p, PIPELINES) for p in cfg.data.test.pipeline
    ]
    test_pipeline = Compose(test_pipeline)

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        output = model(data, mode='test')
    return data, output
