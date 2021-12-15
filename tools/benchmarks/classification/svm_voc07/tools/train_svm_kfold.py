# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
"""SVM training using 3-fold cross-validation.

Relevant transfer tasks: Image Classification VOC07 and COCO2014.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import argparse
import logging
import os
import os.path as osp
import pickle
import sys
import time

import numpy as np
import svm_helper
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from tqdm import tqdm

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def train_svm(opts):
    """Train SVM with k-fold."""
    assert osp.exists(opts.data_file), 'Data file not found. Abort!'
    if not osp.exists(opts.output_path):
        os.makedirs(opts.output_path)

    features, targets = svm_helper.load_input_data(opts.data_file,
                                                   opts.targets_data_file)
    # normalize the features: N x 9216 (example shape)
    features = svm_helper.normalize_features(features)

    # parse the cost values for training the SVM on
    costs_list = svm_helper.parse_cost_list(opts.costs_list)

    # classes for which SVM training should be done
    if opts.cls_list:
        cls_list = [int(cls) for cls in opts.cls_list.split(',')]
    else:
        num_classes = targets.shape[1]
        cls_list = range(num_classes)

    for cls_idx in tqdm(range(len(cls_list))):
        cls = cls_list[cls_idx]
        for cost_idx in range(len(costs_list)):
            start = time.time()
            cost = costs_list[cost_idx]
            out_file, ap_out_file = svm_helper.get_svm_train_output_files(
                cls, cost, opts.output_path)
            if osp.exists(out_file) and osp.exists(ap_out_file):
                logger.info(f'SVM model exists: {out_file}')
                logger.info(f'AP file exists: {ap_out_file}')
            else:
                clf = LinearSVC(
                    C=cost,
                    class_weight={
                        1: 2,
                        -1: 1
                    },
                    intercept_scaling=1.0,
                    verbose=0,
                    penalty='l2',
                    loss='squared_hinge',
                    tol=0.0001,
                    dual=True,
                    max_iter=2000,
                )
                cls_labels = targets[:, cls].astype(dtype=np.int32, copy=True)
                # meaning of labels in VOC/COCO original loaded target files:
                # label 0 = not present, set it to -1 as svm train target
                # label 1 = present. Make the svm train target labels as -1, 1.
                cls_labels[np.where(cls_labels == 0)] = -1

                ap_scores = cross_val_score(
                    clf,
                    features,
                    cls_labels,
                    cv=3,
                    scoring='average_precision')
                clf.fit(features, cls_labels)

                np.save(ap_out_file, np.array([ap_scores.mean()]))
                with open(out_file, 'wb') as fwrite:
                    pickle.dump(clf, fwrite)
            print(f'time: {time.time() - start:.4g} s')


def main():
    parser = argparse.ArgumentParser(description='SVM model training')
    parser.add_argument(
        '--data_file',
        type=str,
        default=None,
        help='Numpy file containing image features')
    parser.add_argument(
        '--targets_data_file',
        type=str,
        default=None,
        help='Numpy file containing image labels')
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='path where to save the trained SVM models')
    parser.add_argument(
        '--costs_list',
        type=str,
        default='0.01,0.1',
        help='comma separated string containing list of costs')
    parser.add_argument(
        '--random_seed',
        type=int,
        default=100,
        help='random seed for SVM classifier training')

    parser.add_argument(
        '--cls_list',
        type=str,
        default=None,
        help='comma separated string list of classes to train')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()
    train_svm(opts)


if __name__ == '__main__':
    main()
