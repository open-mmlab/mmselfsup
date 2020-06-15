import argparse
from mmcv import Config

from openselfsup.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    model = build_model(cfg.model)

    num_params = sum(p.numel() for p in model.parameters()) / 1000000.
    num_grad_params = sum(p.numel() for p in model.parameters() \
        if p.requires_grad) / 1000000.
    num_backbone_params = sum(
        p.numel() for p in model.backbone.parameters()) / 1000000.
    num_backbone_grad_params = sum(p.numel() for p in model.backbone.parameters() \
        if p.requires_grad) / 1000000.
    print(
        "Number of backbone parameters: {:.5g} M".format(num_backbone_params))
    print("Number of backbone parameters requiring grad: {:.5g} M".format(
        num_backbone_grad_params))
    print("Number of total parameters: {:.5g} M".format(num_params))
    print("Number of total parameters requiring grad: {:.5g} M".format(
        num_grad_params))


if __name__ == '__main__':
    main()
