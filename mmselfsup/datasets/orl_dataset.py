# Copyright (c) OpenMMLab. All rights reserved.
import os

import cv2
import mmengine
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from mmselfsup.registry import DATASETS


def to_numpy(pil_img):
    np_img = np.array(pil_img, dtype=np.uint8)
    if np_img.ndim < 3:
        np_img = np.expand_dims(np_img, axis=-1)
    np_img = np.rollaxis(np_img, 2)  # HWC to CHW
    return np_img


def get_max_iou(pred_boxes, gt_box):
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


def correpondence_box_filter(boxes,
                             min_size=20,
                             max_ratio=None,
                             topN=None,
                             max_iou_thr=None):
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


def selective_search(image, method='fast'):
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


def box_filter(boxes, min_size=None, max_ratio=None, topN=None):
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


@DATASETS.register_module()
class SSDataset(Dataset):
    """Dataset for generating selective search proposals."""

    def __init__(self,
                 root,
                 json_file,
                 memcached=False,
                 mclient_path=None,
                 method='fast',
                 min_size=None,
                 max_ratio=None,
                 topN=None):
        data = mmengine.load(json_file)
        self.fns = [item['file_name'] for item in data['images']]
        self.fns = [os.path.join(root, fn) for fn in self.fns]
        self.memcached = memcached
        self.mclient_path = mclient_path
        self.initialized = False
        self.method = method
        self.min_size = min_size
        self.max_ratio = max_ratio
        self.topN = topN

    def __len__(self):
        return len(self.fns)

    def __getitem__(self, idx):
        # if self.memcached:
        #     self._init_memcached()
        # if self.memcached:
        #     img = self.mc_loader(self.fns[idx])
        # else:
        #     img = Image.open(self.fns[idx])
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
            print('=========')
            print('No!')
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

    def get_sample(self, idx):
        # if self.memcached:
        #     self._init_memcached()
        # if self.memcached:
        #     img = self.mc_loader(self.part_fns[idx])
        # else:
        #     img = Image.open(self.part_fns[idx])
        img = Image.open(self.part_fns[idx])
        img = img.convert('RGB')
        # load knn images
        target = self.part_labels[idx][:self.knn_image_num]
        # if self.memcached:
        #     knn_imgs = [self.mc_loader(self.fns[i]) for i in target]
        # else:
        #     knn_imgs = [Image.open(self.fns[i]) for i in target]
        knn_imgs = [Image.open(self.fns[i]) for i in target]
        knn_imgs = [knn_img.convert('RGB') for knn_img in knn_imgs]
        # load selective search proposals
        bbox = self.part_bboxes[idx]
        knn_bboxes = [self.bboxes[i] for i in target]
        return img, knn_imgs, bbox, knn_bboxes


@DATASETS.register_module()
class CorrespondDataset(Dataset):
    """Dataset for generating corresponding intra- and inter-RoIs."""

    def __init__(self,
                 root,
                 knn_json_file,
                 ss_json_file,
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
                 prefetch=False):

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
        self.prefetch = prefetch

    def __len__(self):
        return self.data_source.get_length()

    def __getitem__(self, idx):
        img, knn_imgs, box, knn_boxes = self.data_source.get_sample(idx)
        # print("==================")
        # print(f'knn_imgs: {knn_imgs}')
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
            if self.prefetch:
                patch = torch.from_numpy(to_numpy(patch))
            else:
                # dev 1.x move the format_pipeline to model.datapreprocessor?
                patch = self.format_pipeline(patch)
            patch_list.append(patch)
        knn_patch_lists = []
        for k in range(len(knn_imgs)):
            knn_patch_list = []
            for x, y, w, h in filtered_knn_boxes[k]:
                patch = TF.resized_crop(knn_imgs[k], y, x, h, w,
                                        (self.patch_size, self.patch_size))
                if self.prefetch:
                    patch = torch.from_numpy(to_numpy(patch))
                else:
                    patch = self.format_pipeline(patch)
                    # print(f'patch shape {patch.shape}')
                knn_patch_list.append(patch)
            knn_patch_lists.append(torch.stack(knn_patch_list))
            # print("knn_patch_lists")
            # print(knn_patch_lists)
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

    def evaluate(self, json_file, intra_bbox, inter_bbox, **kwargs):
        assert (len(intra_bbox) == len(inter_bbox)), \
            'Mismatch the number of images in part training set, \
            got: intra: {} inter: {}'\
                .format(len(intra_bbox), len(inter_bbox))
        data = mmengine.load(json_file)
        # dict
        data_new = {}
        # sub-dict
        info = {}
        image_info = {}
        pseudo_anno = {}
        info['bbox_min_size'] = self.min_size
        info['bbox_max_aspect_ratio'] = self.max_ratio
        info['bbox_max_iou'] = self.max_iou_thr
        info['intra_bbox_num'] = self.topN
        info['knn_image_num'] = self.knn_image_num
        info['knn_bbox_pair_ratio'] = self.topk_bbox_ratio
        image_info['file_name'] = data['images']['file_name']
        image_info['id'] = data['images']['id']
        pseudo_anno['image_id'] = data['pseudo_annotations']['image_id']
        pseudo_anno['bbox'] = intra_bbox
        pseudo_anno['knn_image_id'] = data['pseudo_annotations'][
            'knn_image_id']
        pseudo_anno['knn_bbox_pair'] = inter_bbox
        data_new['info'] = info
        data_new['images'] = image_info
        data_new['pseudo_annotations'] = pseudo_anno
        return data_new
