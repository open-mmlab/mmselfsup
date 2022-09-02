# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Optional, Sequence, Union

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
               options: Optional[dict] = None):
    """Initialize a classifier from config file.

    Args:
        config (str or :obj:`mmengine.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): The device where the anchors will be put on.
            Defaults to cuda:0.
        options (dict): Options to override some settings in the used config.

    Returns:
        nn.Module: The initialized model.
    """
    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if options is not None:
        config.merge_from_dict(options)
    config.model.pretrained = None
    config.model.setdefault('data_preprocessor',
                            config.get('data_preprocessor', None))
    model = build_algorithm(config.model)
    if checkpoint is not None:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


ImagesType = Union[str, np.ndarray, Sequence[str], Sequence[np.ndarray]]
SampleList = List[SelfSupDataSample]


def inference_model(model: nn.Module,
                    img: ImagesType) -> Union[SelfSupDataSample, SampleList]:
    """Inference image(s) with the mmselfsup model.

    Args:
        Model (nn.Module): The loaded model.
        imgs (str, ndarray, Sequence[str/ndarray]):
           Either image files or loaded images.
        test_pipeline_cfg (:obj:`Compose`): Test pipeline.

    Returns:
        :obj:`SelfSupDataSample` or list[:obj:`SelfSupDataSample`]:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the model results directly.
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
        results = model.test_step(data)
    return data, results
