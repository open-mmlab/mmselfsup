# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict
from typing import Union

import mmengine
import torch
from mmengine.runner.checkpoint import _load_checkpoint


def convert_mmcls_to_timm(state_dict: Union[OrderedDict, dict]) -> OrderedDict:
    """Convert keys in MMClassification pretrained vit models to timm tyle.

    Args:
        state_dict (Union[OrderedDict, dict]): The state dict of
            MMClassification pretrained vit models.

    Returns:
        OrderedDict: The converted state dict.
    """
    # only keep the backbone weights and remove the backbone. prefix
    state_dict = {
        key.replace('backbone.', ''): value
        for key, value in state_dict.items() if key.startswith('backbone.')
    }

    # replace projection with proj
    state_dict = {
        key.replace('projection', 'proj'): value
        for key, value in state_dict.items()
    }

    # replace ffn.layers.0.0 with mlp.fc1
    state_dict = {
        key.replace('ffn.layers.0.0', 'mlp.fc1'): value
        for key, value in state_dict.items()
    }

    # replace ffn.layers.1 with mlp.fc2
    state_dict = {
        key.replace('ffn.layers.1', 'mlp.fc2'): value
        for key, value in state_dict.items()
    }

    # replace layers with blocks
    state_dict = {
        key.replace('layers', 'blocks'): value
        for key, value in state_dict.items()
    }

    # replace ln with norm
    state_dict = {
        key.replace('ln', 'norm'): value
        for key, value in state_dict.items()
    }

    # replace the last norm1 with norm
    state_dict['norm.weight'] = state_dict.pop('norm1.weight')
    state_dict['norm.bias'] = state_dict.pop('norm1.bias')

    state_dict = OrderedDict({'model': state_dict})
    return state_dict


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in MMClassification '
        'pretrained vit models to timm tyle')
    parser.add_argument('src', help='src model path or url')
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = _load_checkpoint(args.src, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    state_dict = convert_mmcls_to_timm(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(state_dict, args.dst)


if __name__ == '__main__':
    main()
