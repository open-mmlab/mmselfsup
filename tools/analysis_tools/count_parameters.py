# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mmcv import Config

from mmselfsup.models import build_algorithm


def parse_args():
    parser = argparse.ArgumentParser(description='Count model parameters')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    model = build_algorithm(cfg.model)

    num_params = sum(p.numel() for p in model.parameters()) / 1000000.
    num_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad) / 1000000.
    num_backbone_params = sum(p.numel()
                              for p in model.backbone.parameters()) / 1000000.
    num_backbone_grad_params = sum(p.numel()
                                   for p in model.backbone.parameters()
                                   if p.requires_grad) / 1000000.
    print(f'Number of backbone parameters: {num_backbone_params:.5g} M')
    print(f'Number of backbone parameters requiring grad: '
          f'{num_backbone_grad_params:.5g} M')
    print(f'Number of total parameters: {num_params:.5g} M')
    print(f'Number of total parameters requiring grad: '
          f'{num_grad_params:.5g} M')


if __name__ == '__main__':
    main()
