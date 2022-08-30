# Copyright (c) OpenMMLab. All rights reserved.
import os
from unittest import TestCase

import cv2
import numpy as np
import torch
from mmengine.structures import InstanceData
from mmengine.utils import digit_version

from mmselfsup.structures import SelfSupDataSample
from mmselfsup.visualization import SelfSupVisualizer


def _rand_patch_box(num_boxes, h, w):
    cx, cy, bw, bh = torch.rand(num_boxes, 4).T

    if digit_version(torch.__version__) < digit_version('1.7.0'):
        clip = torch.clamp
    else:
        clip = torch.clip

    tl_x = clip(((cx * w) - (w * bw / 2)), 0, w)
    tl_y = clip(((cy * h) - (h * bh / 2)), 0, h)
    br_x = clip(((cx * w) + (w * bw / 2)), 0, w)
    br_y = clip(((cy * h) + (h * bh / 2)), 0, h)

    patch_box = torch.stack([tl_x, tl_y, br_x, br_y]).T
    return patch_box.unsqueeze(0)


class TestSelfSupVisualizer(TestCase):

    def test_add_datasample(self):
        h = 12
        w = 12

        out_file = 'out_file.jpg'

        # ======= test relative_loc =======

        # gt_instances
        num_patch_box = 5
        image = np.random.randint(0, 256, (h, w, 3))
        image = np.expand_dims(image, 0)
        pseudo_label = InstanceData()
        pseudo_label.patch_box = _rand_patch_box(num_patch_box, h, w)
        pseudo_label.unpatched_img = torch.tensor(image)
        gt_selfsup_data_sample = SelfSupDataSample()
        gt_selfsup_data_sample.pseudo_label = pseudo_label

        # pred_instances
        pseudo_label = InstanceData()
        pseudo_label.patch_box = _rand_patch_box(num_patch_box, h, w)
        pseudo_label.unpatched_img = torch.tensor(image)
        pred_selfsup_data_sample = SelfSupDataSample()
        pred_selfsup_data_sample.pseudo_label = pseudo_label

        selfsup_visualizer = SelfSupVisualizer()

        # test gt_instances
        selfsup_visualizer.add_datasample('image', image,
                                          gt_selfsup_data_sample)

        # test out_file
        selfsup_visualizer.add_datasample(
            'image', image, gt_selfsup_data_sample, out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w, 3))

        # test pred_instance
        selfsup_visualizer.add_datasample(
            'image',
            image,
            gt_selfsup_data_sample,
            pred_selfsup_data_sample,
            draw_gt=False,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w, 3))

        # test gt_instances and pred_instances
        selfsup_visualizer.add_datasample(
            'image',
            image,
            gt_selfsup_data_sample,
            pred_selfsup_data_sample,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w * 2, 3))

        # ======= test rotation_pred =======

        # gt_instances
        image = [np.random.randint(0, 256, (h, w, 3)) for _ in range(4)]
        image = np.concatenate(image, axis=1)
        pseudo_label = InstanceData()
        pseudo_label.rot_label = torch.tensor([0, 1, 2, 3])
        gt_selfsup_data_sample = SelfSupDataSample()
        gt_selfsup_data_sample.pseudo_label = pseudo_label

        # pred_instances
        pseudo_label = InstanceData()
        pseudo_label.rot_label = torch.tensor([0, 1, 2, 3])
        pred_selfsup_data_sample = SelfSupDataSample()
        pred_selfsup_data_sample.pseudo_label = pseudo_label

        selfsup_visualizer = SelfSupVisualizer()

        # test gt_instances
        selfsup_visualizer.add_datasample('image', image,
                                          gt_selfsup_data_sample)

        # test out_file
        selfsup_visualizer.add_datasample(
            'image', image, gt_selfsup_data_sample, out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w * 4, 3))

        # test pred_instance
        selfsup_visualizer.add_datasample(
            'image',
            image,
            gt_selfsup_data_sample,
            pred_selfsup_data_sample,
            draw_gt=False,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w * 4, 3))

        # test gt_instances and pred_instances
        selfsup_visualizer.add_datasample(
            'image',
            image,
            gt_selfsup_data_sample,
            pred_selfsup_data_sample,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w * 8, 3))

        # ======= test mask image modeling =======

        # gt_instances
        image = np.random.randint(0, 256, (h, w, 3))
        mask = InstanceData()
        mask.value = torch.tensor([[1, 0], [0, 1]])
        gt_selfsup_data_sample = SelfSupDataSample()
        gt_selfsup_data_sample.mask = mask

        # pred_instances
        mask = InstanceData()
        mask.value = torch.tensor([[1, 0], [0, 1]])
        pred_selfsup_data_sample = SelfSupDataSample()
        pred_selfsup_data_sample.mask = mask

        selfsup_visualizer = SelfSupVisualizer()

        # test gt_instances
        selfsup_visualizer.add_datasample('image', image,
                                          gt_selfsup_data_sample)

        # test out_file
        selfsup_visualizer.add_datasample(
            'image', image, gt_selfsup_data_sample, out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w, 3))

        # test pred_instance
        selfsup_visualizer.add_datasample(
            'image',
            image,
            gt_selfsup_data_sample,
            pred_selfsup_data_sample,
            draw_gt=False,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w, 3))

        # test gt_instances and pred_instances
        selfsup_visualizer.add_datasample(
            'image',
            image,
            gt_selfsup_data_sample,
            pred_selfsup_data_sample,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w * 2, 3))

        # ======= test contrastive learning =======
        # gt_instances
        image = [np.random.randint(0, 256, (h, w, 3)) for _ in range(2)]
        image = np.concatenate(image, axis=1)
        gt_selfsup_data_sample = SelfSupDataSample()

        # pred_instances
        pred_selfsup_data_sample = SelfSupDataSample()

        selfsup_visualizer = SelfSupVisualizer()

        # test gt_instances
        selfsup_visualizer.add_datasample('image', image,
                                          gt_selfsup_data_sample)

        # test out_file
        selfsup_visualizer.add_datasample(
            'image', image, gt_selfsup_data_sample, out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w * 2, 3))

        # test pred_instance
        selfsup_visualizer.add_datasample(
            'image',
            image,
            gt_selfsup_data_sample,
            pred_selfsup_data_sample,
            draw_gt=False,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w * 2, 3))

        # test gt_instances and pred_instances
        selfsup_visualizer.add_datasample(
            'image',
            image,
            gt_selfsup_data_sample,
            pred_selfsup_data_sample,
            out_file=out_file)
        self._assert_image_and_shape(out_file, (h, w * 2 * 2, 3))

    def _assert_image_and_shape(self, out_file, out_shape):
        assert os.path.exists(out_file)
        drawn_img = cv2.imread(out_file)
        assert drawn_img.shape == out_shape
        os.remove(out_file)
