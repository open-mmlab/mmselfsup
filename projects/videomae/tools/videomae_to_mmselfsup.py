# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader


def convert_videomae(ckpt):
    new_ckpt = OrderedDict()

    for k, v in list(ckpt.items()):
        new_v = v
        if k.startswith('encoder.'):
            new_k = k.replace('encoder.', 'backbone.')
        elif k.startswith('decoder'):
            new_k = k.replace('decoder.', 'neck.')
        elif k.startswith('encoder_to_decoder'):
            new_k = k.replace('encoder_to_decoder.',
                              'neck.decoder_embed_layer.')
        elif k.startswith('mask_token'):
            new_k = 'neck.' + k

        # second round
        if 'patch_embed.proj.' in new_k:
            new_k = new_k.replace('patch_embed.proj.',
                                  'patch_embed.projection.')

        if 'mlp.fc1' in new_k:
            new_k = new_k.replace('mlp.fc1', 'mlp.layers.0.0')

        if 'mlp.fc2' in new_k:
            new_k = new_k.replace('mlp.fc2', 'mlp.layers.1')

        new_ckpt[new_k] = new_v
    return new_ckpt


def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in pretrained clip models to mmcls style.')
    parser.add_argument('src', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst', help='save path')
    args = parser.parse_args()

    checkpoint = CheckpointLoader.load_checkpoint(args.src, map_location='cpu')

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    weight = convert_videomae(state_dict)
    mmengine.mkdir_or_exist(osp.dirname(args.dst))
    torch.save(weight, args.dst)

    print('Done!!')


if __name__ == '__main__':
    main()
