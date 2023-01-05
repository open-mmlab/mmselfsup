# Copyright (c) OpenMMLab. All rights reserved.
<<<<<<< HEAD

from typing import Optional, Union

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose, default_collate
from mmengine.runner import load_checkpoint
from torch import nn

from mmselfsup.models import build_algorithm
from mmselfsup.structures import SelfSupDataSample


def init_model(config: Union[str, Config],
               checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               options: Optional[dict] = None) -> nn.Module:
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): The device where the anchors will be put on.
            Defaults to cuda:0.
        options (dict): Options to override some settings in the used config.

=======
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
>>>>>>> upstream/master
    Returns:
        nn.Module: The initialized model.
    """
    if isinstance(config, str):
<<<<<<< HEAD
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
=======
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
>>>>>>> upstream/master
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if options is not None:
        config.merge_from_dict(options)
<<<<<<< HEAD
    config.model.pretrained = None
    config.model.setdefault('data_preprocessor',
                            config.get('data_preprocessor', None))
=======
>>>>>>> upstream/master
    model = build_algorithm(config.model)
    if checkpoint is not None:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


<<<<<<< HEAD
def inference_model(model: nn.Module,
                    img: Union[str, np.ndarray]) -> SelfSupDataSample:
    """Inference an image with the mmselfsup model.

    Args:
        model (nn.Module): The loaded model.
        img (Union[str, ndarray]):
           Either image path or loaded image.

    Returns:
        SelfSupDataSample: Output of model inference.
    """
    cfg = model.cfg
    # build the data pipeline
    test_pipeline_cfg = cfg.test_dataloader.dataset.pipeline
    if isinstance(img, str):
        if test_pipeline_cfg[0]['type'] != 'LoadImageFromFile':
            test_pipeline_cfg.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_path=img)
    else:
        if test_pipeline_cfg[0]['type'] == 'LoadImageFromFile':
            test_pipeline_cfg.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(test_pipeline_cfg)
    data = test_pipeline(data)
    data = default_collate([data])

    # forward the model
    with torch.no_grad():
        inputs, data_samples = model.data_preprocessor(data, False)
        features = model(inputs, data_samples, mode='tensor')
    return features
=======
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
>>>>>>> upstream/master
