# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified from https://colab.research.google.com/github/facebookresearch/mae
# /blob/main/demo/mae_visualize.ipynb
from argparse import ArgumentParser
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.dataset import Compose, default_collate

from mmselfsup.apis import inference_model, init_model
from mmselfsup.utils import register_all_modules

# imagenet_mean = np.array([0.485, 0.456, 0.406])
# imagenet_std = np.array([0.229, 0.224, 0.225])
imagenet_mean = np.array([123.675, 116.28, 103.53]),
imagenet_std = np.array([58.395, 57.12, 57.375]),


def show_image(img: torch.Tensor, title: str = '') -> None:
    # image is [H, W, 3]
    assert img.shape[2] == 3

    plt.imshow(img)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def save_images(x: torch.Tensor, img_masked: torch.Tensor, y: torch.Tensor,
                img_paste: torch.Tensor, out_file: str) -> None:
    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 6]

    plt.subplot(1, 4, 1)
    show_image(x, 'original')

    plt.subplot(1, 4, 2)
    show_image(img_masked, 'masked')

    plt.subplot(1, 4, 3)
    show_image(y, 'reconstruction')

    plt.subplot(1, 4, 4)
    show_image(img_paste, 'reconstruction + visible')

    plt.savefig(out_file)
    print(f'Images are saved to {out_file}')


def recover_norm(img: torch.Tensor,
                 mean: np.ndarray = np.array([0.485, 0.456, 0.406]),
                 std: np.ndarray = np.array([0.229, 0.224, 0.225])):
    img = torch.clip((img * std + mean) * 255, 0, 255).int()
    return img


def post_process(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # channel conversion
    x = torch.einsum('nchw->nhwc', x.cpu())
    # masked image
    img_masked = x * (1 - mask)
    # reconstructed image pasted with visible patches
    img_paste = x * (1 - mask) + y * mask

    # muptiply std and add mean to each image
    x = recover_norm(x[0])
    img_masked = recover_norm(img_masked[0])
    y = recover_norm(y[0])
    img_paste = recover_norm(img_paste[0])

    return x, img_masked, y, img_paste


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Model config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--img-path', help='Image file path')
    parser.add_argument('--out-file', help='The output image file path')
    parser.add_argument(
        '--use-vis-pipeline',
        action='store_true',
        help='Use vis_pipeline defines in config. For some algorithms, like '
        'SimMIM, they generate mask in data pipeline, thus apply pipeline in '
        'config to obtain the mask.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    print('Model loaded.')

    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(0)
    print('Pixel reconstruction.')

    if args.use_vis_pipeline:
        model.cfg.test_dataloader = dict(
            dataset=dict(pipeline=model.cfg.vis_pipeline))
    else:
        model.cfg.test_dataloader = dict(
            dataset=dict(pipeline=[
                dict(
                    type='LoadImageFromFile',
                    file_client_args=dict(backend='disk')),
                dict(type='Resize', scale=(224, 224), backend='pillow'),
                dict(type='PackSelfSupInputs', meta_keys=['img_path'])
            ]))

    # get original image
    vis_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    data = dict(img_path=args.img_path)
    data = vis_pipeline(data)
    data = default_collate([data])
    img, _ = model.data_preprocessor(data, False)

    # get reconstruction image
    features = inference_model(model, args.img_path)
    results = model.reconstruct(features)

    x, img_masked, y, img_paste = post_process(img[0], results.pred.value,
                                               results.mask.value)

    save_images(x, img_masked, y, img_paste, args.out_file)


if __name__ == '__main__':
    main()
