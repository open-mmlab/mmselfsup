# Copyright (c) OpenMMLab. All rights reserved.
import math
import os
import random
from typing import Union

import cv2
import mmengine
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from mmengine import build_from_cfg
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from mmselfsup.registry import DATASETS, TRANSFORMS


def get_max_iou(pred_boxes: list, gt_box: list) -> np.float:
    """
    pred_boxes : multiple coordinate for predict bounding boxes (x, y, w, h)
    gt_box :   the coordinate for ground truth bounding box (x, y, w, h)
    return :   the max iou score about pred_boxes and gt_box
    """
    # 1.get the coordinate of inters
    ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
    ixmax = np.minimum(pred_boxes[:, 0] + pred_boxes[:, 2],
                       gt_box[0] + gt_box[2])
    iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
    iymax = np.minimum(pred_boxes[:, 1] + pred_boxes[:, 3],
                       gt_box[1] + gt_box[3])

    iw = np.maximum(ixmax - ixmin, 0.)
    ih = np.maximum(iymax - iymin, 0.)

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of union
    uni = (
        pred_boxes[:, 2] * pred_boxes[:, 3] + gt_box[2] * gt_box[3] - inters)

    # 4. calculate the overlaps and find the max overlap
    #   between pred_boxes and gt_box
    iou = inters / uni
    iou_max = np.max(iou)

    return iou_max


def correpondence_box_filter(boxes: list,
                             min_size=20,
                             max_ratio=None,
                             topN=None,
                             max_iou_thr=None) -> Union[list, None]:
    proposal = []

    for box in boxes:
        # Calculate width and height of the box
        w, h = box[2], box[3]

        # Filter for size
        if min_size:
            if w < min_size or h < min_size:
                continue

        # Filter for box ratio
        if max_ratio:
            if w / h > max_ratio or h / w > max_ratio:
                continue

        # Filter for overlap
        if max_iou_thr:
            if len(proposal):
                iou_max = get_max_iou(np.array(proposal), np.array(box))
                if iou_max > max_iou_thr:
                    continue

        proposal.append(box)

    if not len(proposal):  # ensure at least one box for each image
        proposal.append(boxes[0])

    if topN:
        if topN <= len(proposal):
            return proposal[:topN]
        else:
            return proposal
    else:
        return


def selective_search(image, method='fast') -> list:
    # initialize OpenCV's selective search implementation
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # set the input image
    ss.setBaseImage(image)
    # check to see if we are using the *fast* but *less accurate* version
    # of selective search
    if method == 'fast':
        # print("[INFO] using *fast* selective search")
        ss.switchToSelectiveSearchFast()
    # otherwise we are using the *slower* but *more accurate* version
    else:
        # print("[INFO] using *quality* selective search")
        ss.switchToSelectiveSearchQuality()
    # run selective search on the input image
    boxes = ss.process()
    return boxes


def box_filter(boxes: list, min_size=None, max_ratio=None, topN=None) -> list:
    proposal = []

    for box in boxes:
        # Calculate width and height of the box
        w, h = box[2], box[3]

        # Filter for size
        if min_size:
            if w < min_size or h < min_size:
                continue

        # Filter for box ratio
        if max_ratio:
            if w / h > max_ratio or h / w > max_ratio:
                continue

        proposal.append(box)

    if topN:
        if topN <= len(proposal):
            return proposal[:topN]
        else:
            return proposal
    else:
        return proposal


def get_iou(pred_box: list, gt_box: list) -> float:
    """
    pred_box : the coordinate for predict bounding box (x, y, w, h)
    gt_box :   the coordinate for ground truth bounding box (x, y, w, h)
    return :   the iou score
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[0] + pred_box[2], gt_box[0] + gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[1] + pred_box[3], gt_box[1] + gt_box[3])

    iw = max(ixmax - ixmin, 0.)
    ih = max(iymax - iymin, 0.)

    # 2. calculate the area of inters
    inters = iw * ih

    # 3. calculate the area of union
    uni = (pred_box[2] * pred_box[3] + gt_box[2] * gt_box[3] - inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / float(uni)

    return iou


def aug_bbox(img,
             box: list,
             shift: tuple,
             scale: tuple,
             ratio: tuple,
             iou_thr: float,
             attempt_num=200) -> list:
    img_w, img_h = img.size
    x, y, w, h = box[0], box[1], box[2], box[3]
    cx, cy = (x + 0.5 * w), (y + 0.5 * h)
    area = w * h
    for attempt in range(attempt_num):
        aug_area = random.uniform(*scale) * area
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aug_ratio = math.exp(random.uniform(*log_ratio))
        aug_w = int(round(math.sqrt(aug_area * aug_ratio)))
        aug_h = int(round(math.sqrt(aug_area / aug_ratio)))
        aug_cx = cx + random.uniform(*shift) * w
        aug_cy = cy + random.uniform(*shift) * h
        aug_x, aug_y = int(round(aug_cx - 0.5 * aug_w)), int(
            round(aug_cy - 0.5 * aug_h))
        if aug_x >= 0 and aug_y >= 0 and (aug_x + aug_w) <= img_w and (
                aug_y + aug_h) <= img_h:
            aug_box = [aug_x, aug_y, aug_w, aug_h]
            if iou_thr is not None:
                iou = get_iou(aug_box, box)
                if iou > iou_thr:
                    return aug_box
            else:
                return aug_box
    return box


@DATASETS.register_module()
class SSDataset(Dataset):
    """Dataset for generating selective search proposals."""

    def __init__(self,
                 root: str,
                 json_file: str,
                 method='fast',
                 min_size=None,
                 max_ratio=None,
                 topN=None):
        data = mmengine.load(json_file)
        self.fns = [item['file_name'] for item in data['images']]
        self.fns = [os.path.join(root, fn) for fn in self.fns]
        self.initialized = False
        self.method = method
        self.min_size = min_size
        self.max_ratio = max_ratio
        self.topN = topN

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.fns[idx])
        img = img.convert('RGB')
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        boxes = selective_search(img_cv2, self.method)
        if self.topN is not None:
            boxes = box_filter(boxes, self.min_size, self.max_ratio, self.topN)
        boxes = torch.from_numpy(np.array(boxes))
        # bbox: Bx4
        # B is the total number of original/topN selective search bboxes
        return dict(bbox=boxes)


class CorrespondJson(object):

    def __init__(
        self,
        root: str,
        knn_json_file: str,
        ss_json_file: str,
        knn_image_num: int,
        part=0,
        num_parts=1,
        data_len=118287,
    ):
        assert part in np.arange(num_parts).tolist(), \
            'part order must be within [0, num_parts)'

        print('loading knn json file...')
        data = mmengine.load(knn_json_file)
        print('loaded knn json file!')
        print('loading selective search json file, this may take minutes...')
        if isinstance(ss_json_file, list):
            data_ss_list = [mmengine.load(ss) for ss in ss_json_file]
            self.bboxes = []
            for ss in data_ss_list:
                self.bboxes += ss['bbox']
        else:
            data_ss = mmengine.load(ss_json_file)
            self.bboxes = data_ss['bbox']
        print('loaded selective search json file!')
        # divide the whole dataset into several parts
        # to enable parallel roi pair retrieval.
        # each part should be run on single gpu
        # and all parts can be run on multiple gpus in parallel.
        part_len = int(data_len / num_parts)
        print('processing part {}...'.format(part))
        if part == num_parts - 1:  # last part
            self.fns = data['images']['file_name']
            self.fns = [os.path.join(root, fn) for fn in self.fns]
            self.part_fns = self.fns[part * part_len:]
            self.part_labels = data['pseudo_annotations']['knn_image_id']
            self.part_bboxes = self.bboxes
        else:
            self.fns = data['images']['file_name']
            self.fns = [os.path.join(root, fn) for fn in self.fns]
            self.part_fns = self.fns[part * part_len:(part + 1) * part_len]
            self.part_labels = data['pseudo_annotations']['knn_image_id'][
                part * part_len:(part + 1) * part_len]
            self.part_bboxes = self.bboxes[part * part_len:(part + 1) *
                                           part_len]
        self.knn_image_num = knn_image_num

    def get_length(self):
        return len(self.part_fns)

    def get_sample(self, idx: int):
        img = Image.open(self.part_fns[idx])
        img = img.convert('RGB')
        # load knn images
        target = self.part_labels[idx][:self.knn_image_num]
        knn_imgs = [Image.open(self.fns[i]) for i in target]
        knn_imgs = [knn_img.convert('RGB') for knn_img in knn_imgs]
        # load selective search proposals
        bbox = self.part_bboxes[idx]
        knn_bboxes = [self.bboxes[i] for i in target]
        return img, knn_imgs, bbox, knn_bboxes


@DATASETS.register_module()
class CorrespondDataset(Dataset):
    """Dataset for generating corresponding intra- and inter-RoIs."""

    def __init__(
        self,
        root: str,
        knn_json_file: str,
        ss_json_file: str,
        part=0,
        num_parts=1,
        data_len=118287,
        norm_cfg=None,
        patch_size=224,
        min_size=96,
        max_ratio=3,
        topN=100,
        max_iou_thr=0.5,
        knn_image_num=10,
        topk_bbox_ratio=0.1,
    ):

        self.data_source = CorrespondJson(root, knn_json_file, ss_json_file,
                                          knn_image_num, part, num_parts,
                                          data_len)
        self.format_pipeline = Compose([
            transforms.ToTensor(),
            transforms.Normalize(norm_cfg['mean'], norm_cfg['std'])
        ])
        self.patch_size = patch_size
        self.min_size = min_size
        self.max_ratio = max_ratio
        self.topN = topN
        self.max_iou_thr = max_iou_thr
        self.knn_image_num = knn_image_num
        self.topk_bbox_ratio = topk_bbox_ratio

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx: int) -> dict:
        img, knn_imgs, box, knn_boxes = self.data_source.get_sample(idx)
        filtered_box = correpondence_box_filter(box, self.min_size,
                                                self.max_ratio, self.topN,
                                                self.max_iou_thr)
        filtered_knn_boxes = [
            correpondence_box_filter(knn_box, self.min_size, self.max_ratio,
                                     self.topN, self.max_iou_thr)
            for knn_box in knn_boxes
        ]
        patch_list = []
        for x, y, w, h in filtered_box:
            patch = TF.resized_crop(img, y, x, h, w,
                                    (self.patch_size, self.patch_size))
            patch = self.format_pipeline(patch)
            patch_list.append(patch)
        knn_patch_lists = []
        for k in range(len(knn_imgs)):
            knn_patch_list = []
            for x, y, w, h in filtered_knn_boxes[k]:
                patch = TF.resized_crop(knn_imgs[k], y, x, h, w,
                                        (self.patch_size, self.patch_size))
                patch = self.format_pipeline(patch)
                knn_patch_list.append(patch)
            knn_patch_lists.append(torch.stack(knn_patch_list))

        filtered_box = torch.from_numpy(np.array(filtered_box))
        filtered_knn_boxes = [
            torch.from_numpy(np.array(knn_box))
            for knn_box in filtered_knn_boxes
        ]
        knn_img_keys = ['{}nn_img'.format(k) for k in range(len(knn_imgs))]
        knn_bbox_keys = ['{}nn_bbox'.format(k) for k in range(len(knn_imgs))]
        # img: BCHW, knn_img: K BCHW, bbox: Bx4, knn_bbox= K Bx4
        # K is the number of knn images, B is the number of filtered bboxes
        dict1 = dict(img=torch.stack(patch_list))
        dict2 = dict(bbox=filtered_box)
        dict3 = dict(img_keys=dict(zip(knn_img_keys, knn_patch_lists)))
        dict4 = dict(bbox_keys=dict(zip(knn_bbox_keys, filtered_knn_boxes)))
        return {**dict1, **dict2, **dict3, **dict4}


class COCOORLJson(object):

    def __init__(self, root: str, json_file: str, topk_knn_image: int):
        data = mmengine.load(json_file)
        self.fns = data['images']['file_name']
        self.intra_bboxes = data['pseudo_annotations']['bbox']
        self.total_knn_image_num = data['info']['knn_image_num']
        self.knn_image_ids = data['pseudo_annotations']['knn_image_id']
        self.knn_bbox_pairs = data['pseudo_annotations'][
            'knn_bbox_pair']  # NxKx(topk_bbox_num)x8
        self.fns = [os.path.join(root, fn) for fn in self.fns]
        self.topk_knn_image = topk_knn_image
        assert self.topk_knn_image <= self.total_knn_image_num, \
            'Top-k knn image number exceeds total number of knn images!'

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        # randomly select one knn image
        rnd = random.randint(0, self.topk_knn_image - 1)
        target_id = self.knn_image_ids[idx][rnd]
        img = Image.open(self.fns[idx])
        knn_img = Image.open(self.fns[target_id])
        img = img.convert('RGB')
        knn_img = knn_img.convert('RGB')
        # load proposals
        intra_bbox = self.intra_bboxes[idx]
        knn_bbox = self.knn_bbox_pairs[idx][rnd]  # (topk_bbox_num)x8
        return img, knn_img, intra_bbox, knn_bbox


@DATASETS.register_module()
class ORLDataset(Dataset):
    """Dataset for ORL."""

    def __init__(
            self,
            root: str,
            json_file: str,
            topk_knn_image: int,
            img_norm_cfg: dict,
            # data_source,
            img_pipeline1: list,
            img_pipeline2: list,
            patch_pipeline1: list,
            patch_pipeline2: list,
            patch_size=224,
            interpolation: int = 2,
            shift=(-0.5, 0.5),
            scale=(0.5, 2.),
            ratio=(0.5, 2.),
            iou_thr=0.5,
            attempt_num=200,
            prefetch=False):
        self.data_source = COCOORLJson(root, json_file, topk_knn_image)
        self.format_pipeline = Compose([
            transforms.ToTensor(),
            # transforms.Normalize(img_norm_cfg['mean'], img_norm_cfg['std'])
        ])
        img_pipeline1 = [build_from_cfg(p, TRANSFORMS) for p in img_pipeline1]
        img_pipeline2 = [build_from_cfg(p, TRANSFORMS) for p in img_pipeline2]
        patch_pipeline1 = [
            build_from_cfg(p, TRANSFORMS) for p in patch_pipeline1
        ]
        patch_pipeline2 = [
            build_from_cfg(p, TRANSFORMS) for p in patch_pipeline2
        ]
        self.img_pipeline1 = Compose(img_pipeline1)
        self.img_pipeline2 = Compose(img_pipeline2)
        self.patch_pipeline1 = Compose(patch_pipeline1)
        self.patch_pipeline2 = Compose(patch_pipeline2)
        self.patch_size = patch_size
        self.interpolation = interpolation
        self.shift = shift
        self.scale = scale
        self.ratio = ratio
        self.iou_thr = iou_thr
        self.attempt_num = attempt_num
        self.prefetch = prefetch

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx: list) -> dict:
        img, knn_img, intra_box, knn_box = self.data_source.get_sample(idx)
        ibox1 = random.choice(intra_box)
        ibox2 = aug_bbox(img, ibox1, self.shift, self.scale, self.ratio,
                         self.iou_thr, self.attempt_num)
        kbox_pair = random.choice(knn_box)
        kbox1, kbox2 = kbox_pair[:4], kbox_pair[4:]
        ipatch1 = TF.resized_crop(
            img,
            ibox1[1],
            ibox1[0],
            ibox1[3],
            ibox1[2], (self.patch_size, self.patch_size),
            interpolation=self.interpolation)
        ipatch2 = TF.resized_crop(
            img,
            ibox2[1],
            ibox2[0],
            ibox2[3],
            ibox2[2], (self.patch_size, self.patch_size),
            interpolation=self.interpolation)
        kpatch1 = TF.resized_crop(
            img,
            kbox1[1],
            kbox1[0],
            kbox1[3],
            kbox1[2], (self.patch_size, self.patch_size),
            interpolation=self.interpolation)
        kpatch2 = TF.resized_crop(
            knn_img,
            kbox2[1],
            kbox2[0],
            kbox2[3],
            kbox2[2], (self.patch_size, self.patch_size),
            interpolation=self.interpolation)
        img1 = self.img_pipeline1({'img': np.array(img)})
        img2 = self.img_pipeline2({'img': np.array(img)})
        ipatch1 = self.patch_pipeline1({'img': np.array(ipatch1)})
        ipatch2 = self.patch_pipeline2({'img': np.array(ipatch2)})
        kpatch1 = self.patch_pipeline1({'img': np.array(kpatch1)})
        kpatch2 = self.patch_pipeline2({'img': np.array(kpatch2)})
        img1 = self.format_pipeline(img1['img'])
        img2 = self.format_pipeline(img2['img'])
        ipatch1 = self.format_pipeline(ipatch1['img'])
        ipatch2 = self.format_pipeline(ipatch2['img'])
        kpatch1 = self.format_pipeline(kpatch1['img'])
        kpatch2 = self.format_pipeline(kpatch2['img'])

        assert img1.shape[0] == 3
        img_cat = torch.cat((img1.unsqueeze(0), img2.unsqueeze(0)), dim=0)
        ipatch_cat = torch.cat((ipatch1.unsqueeze(0), ipatch2.unsqueeze(0)),
                               dim=0)
        kpatch_cat = torch.cat((kpatch1.unsqueeze(0), kpatch2.unsqueeze(0)),
                               dim=0)
        return dict(img=[img_cat, ipatch_cat, kpatch_cat], sample_idx=idx)
