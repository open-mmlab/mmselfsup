# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified from https://colab.research.google.com/github/facebookresearch/mae
# /blob/main/demo/mae_visualize.ipynb

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from mmselfsup.apis import inference_model, init_model

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    image = torch.clip((image * imagenet_std + imagenet_mean) * 255, 0,
                       255).int()
    plt.imshow(image)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def show_images(x, im_masked, y, im_paste):
    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 6]

    plt.subplot(1, 4, 1)
    show_image(x, 'original')

    plt.subplot(1, 4, 2)
    show_image(im_masked, 'masked')

    plt.subplot(1, 4, 3)
    show_image(y, 'reconstruction')

    plt.subplot(1, 4, 4)
    show_image(im_paste, 'reconstruction + visible')

    plt.show()


def post_process(x, y, mask):
    x = torch.einsum('nchw->nhwc', x.cpu())
    # masked image
    im_masked = x * (1 - mask)
    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask
    return x[0], im_masked[0], y[0], im_paste[0]


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='MAE Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Whether to show the predict results by matplotlib.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    print('Model loaded.')

    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    print('MAE with pixel reconstruction:')

    img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model.cfg.data = dict(
        test=dict(pipeline=[
            dict(type='Resize', size=(224, 224)),
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
        ]))

    img = Image.open(args.img)
    img, (mask, pred) = inference_model(model, img)
    x, im_masked, y, im_paste = post_process(img, pred, mask)
    if args.show:
        show_images(x, im_masked, y, im_paste)


if __name__ == '__main__':
    main()
