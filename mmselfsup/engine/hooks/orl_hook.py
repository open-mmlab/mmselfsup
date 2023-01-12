# Copyright (c) OpenMMLab. All rights reserved.
import json
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from mmengine.dist import is_distributed
from mmengine.hooks import Hook
from mmengine.logging import print_log

from mmselfsup.models.utils import Extractor
from mmselfsup.registry import HOOKS


@HOOKS.register_module()
class ORLHook(Hook):
    """ORL feature extractor hook."""

    def __init__(self,
                 extract_dataloader: dict,
                 keys: int = 10,
                 normalize=True,
                 seed: Optional[int] = None) -> None:

        self.dist_mode = is_distributed()
        self.keys = keys
        self.knn_batchsize = extract_dataloader['batch_size']
        self.dataset = extract_dataloader['dataset']
        self.extractor = Extractor(
            extract_dataloader=extract_dataloader,
            seed=seed,
            dist_mode=self.dist_mode,
            pool_cfg=None)
        self.normalize = normalize

    def retrieve_knn(self, features: torch.Tensor):
        """
        retrieve knn image ids for each image in COCO train2017
        Args:
        features : Embeddings of all images in COCO train 2017
        """
        # load data
        data_root = self.dataset['data_root']
        data_ann = self.dataset['ann_file']
        data_prefix = self.dataset['data_prefix']['img']
        train_json = data_root + data_ann
        train_root = data_root + data_prefix
        # train_json = '../data/coco/annotations/instances_train2017.json'
        # train_root = '../data/coco/train2017/'
        with open(train_json, 'r') as json_file:
            data = json.load(json_file)

        train_fns = [train_root + item['file_name'] for item in data['images']]
        imgids = [item['id'] for item in data['images']]
        knn_imgids = []
        # batch processing
        batch = self.knn_batchsize
        keys = self.keys

        feat_bank = features
        for i in range(0, len(train_fns), batch):
            print('[INFO] processing batch: {}'.format(i + 1))
            start = time.time()
            if (i + batch) < len(train_fns):
                query_feats = feat_bank[i:i + batch, :]
            else:
                query_feats = feat_bank[i:len(train_fns), :]
            similarity = torch.mm(query_feats, feat_bank.T)
            I_knn = torch.topk(similarity, keys + 1, dim=1)[1].cpu()
            I_knn = I_knn[:, 1:]  # exclude itself (i.e., 1st nn)
            knn_list = I_knn.numpy().tolist()
            [knn_imgids.append(knn) for knn in knn_list]
            end = time.time()
            print('[INFO] batch {} took {:.4f} seconds'.format(
                i + 1, end - start))

        # 118287 for coco, 241690 for coco+
        num_image = len(train_fns)
        save_dir = data_root + '/meta/'
        save_path = save_dir + 'train2017_{}nn_instance.json'.format(keys)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        assert len(imgids) == len(knn_imgids) == len(train_fns) == num_image, \
            f'Mismatch number of training images, got: {len(knn_imgids)}'
        # dict
        data_new = {}
        info = {}
        image_info = {}
        pseudo_anno = {}
        info['knn_image_num'] = keys
        image_info['file_name'] = [
            item['file_name'] for item in data['images']
        ]
        image_info['id'] = [item['id'] for item in data['images']]
        pseudo_anno['image_id'] = imgids
        pseudo_anno['knn_image_id'] = knn_imgids
        data_new['info'] = info
        data_new['images'] = image_info
        data_new['pseudo_annotations'] = pseudo_anno
        with open(save_path, 'w') as f:
            json.dump(data_new, f)
        print('[INFO] image-level knn json file has been saved to {}'.format(
            save_path))

    def after_run(self, runner):
        self._extract_func(runner)

    def _extract_func(self, runner):
        # step 1: get features
        runner.model.eval()
        features = self.extractor(runner.model.module)

        # step 2: save features
        if not self.dist_mode or (self.dist_mode and runner.rank == 0):
            if self.normalize:
                features = nn.functional.normalize(features['feat'], dim=1)
            np.save(
                '{}/feature_epoch_{}.npy'.format(runner.work_dir,
                                                 runner.epoch),
                features.cpu().numpy())
            print_log(
                'Feature extraction done!!! total features: {}\t\
                feature dimension: {}'.format(
                    features.size(0), features.size(1)),
                logger='current')

        # step3: retrieval knn
        if runner.rank == 0:
            self.retrieve_knn(features)
