# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://colab.research.google.com/github/facebookresearch/mae
# /blob/main/demo/mae_visualize.ipynb
import random
from argparse import ArgumentParser
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmengine.dataset import Compose, default_collate

from mmselfsup.apis import inference_model, init_model
from mmselfsup.utils import register_all_modules

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def show_image(img: torch.Tensor, title: str = '') -> None:
    # image is [H, W, 3]
    assert img.shape[2] == 3

    plt.imshow(img)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def save_images(original_img: torch.Tensor, img_masked: torch.Tensor,
                pred_img: torch.Tensor, img_paste: torch.Tensor,
                out_file: str) -> None:
    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 6]

    plt.subplot(1, 4, 1)
    show_image(original_img, 'original')

    plt.subplot(1, 4, 2)
    show_image(img_masked, 'masked')

    plt.subplot(1, 4, 3)
    show_image(pred_img, 'reconstruction')

    plt.subplot(1, 4, 4)
    show_image(img_paste, 'reconstruction + visible')

    plt.savefig(out_file)
    print(f'Images are saved to {out_file}')


def recover_norm(img: torch.Tensor,
                 mean: np.ndarray = imagenet_mean,
                 std: np.ndarray = imagenet_std):
    if mean is not None and std is not None:
        img = torch.clip((img * std + mean) * 255, 0, 255).int()
    return img


def post_process(
    original_img: torch.Tensor,
    pred_img: torch.Tensor,
    mask: torch.Tensor,
    mean: np.ndarray = imagenet_mean,
    std: np.ndarray = imagenet_std
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # channel conversion
    original_img = torch.einsum('nchw->nhwc', original_img.cpu())
    # masked image
    img_masked = original_img * (1 - mask)
    # reconstructed image pasted with visible patches
    img_paste = original_img * (1 - mask) + pred_img * mask

    # muptiply std and add mean to each image
    original_img = recover_norm(original_img[0])
    img_masked = recover_norm(img_masked[0])

    pred_img = recover_norm(pred_img[0])
    img_paste = recover_norm(img_paste[0])

    return original_img, img_masked, pred_img, img_paste


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Model config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument('--img-path', help='Image file path')
    parser.add_argument('--out-file', help='The output image file path')
    parser.add_argument(
        '--use-vis-pipeline',
        action='store_true',
        help='Use vis_pipeline defined in config. For some algorithms, such '
        'as SimMIM and MaskFeat, they generate mask in data pipeline, thus '
        'the visualization process applies vis_pipeline in config to obtain '
        'the mask.')
    parser.add_argument(
        '--norm-pix',
        action='store_true',
        help='MAE uses `norm_pix_loss` for optimization in pre-training, thus '
        'the visualization process also need to compute mean and std of each '
        'patch embedding while reconstructing the original images.')
    parser.add_argument(
        '--target-generator',
        action='store_true',
        help='Some algorithms use target_generator for optimization in '
        'pre-training, such as MaskFeat, thus the visualization process could '
        'turn this on to visualize the target instead of RGB image.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='The random seed for visualization')
    args = parser.parse_args()

    register_all_modules()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    print('Model loaded.')

    # make random mask reproducible (comment out to make it change)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print('Reconstruction visualization.')

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

    if args.norm_pix:
        # for MAE reconstruction
        img_embedding = model.head.patchify(img[0])
        # normalize the target image
        mean = img_embedding.mean(dim=-1, keepdim=True)
        std = (img_embedding.var(dim=-1, keepdim=True) + 1.e-6)**.5
    else:
        mean = imagenet_mean
        std = imagenet_std

    # get reconstruction image
    features = inference_model(model, args.img_path)
    results = model.reconstruct(features, mean=mean, std=std)

    original_target = model.target if args.target_generator else img[0]

    original_img, img_masked, pred_img, img_paste = post_process(
        original_target,
        results.pred.value,
        results.mask.value,
        mean=mean,
        std=std)

    save_images(original_img, img_masked, pred_img, img_paste, args.out_file)


if __name__ == '__main__':
    main()
