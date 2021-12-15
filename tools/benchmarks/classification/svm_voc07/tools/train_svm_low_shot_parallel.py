# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
"""Low Shot SVM training.

Relevant transfer tasks: Low-shot Image Classification VOC07 and Places205 low
shot samples.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import argparse
import multiprocessing as mp
import os
import os.path as osp
import pickle
import sys

import svm_helper
import tqdm
from sklearn.svm import LinearSVC


def task(cls, cost, opts, features, targets):
    """The task function to train the model."""
    suffix = '_'.join(
        opts.targets_data_file.split('/')[-1].split('.')[0].split('_')[-2:])
    out_file = svm_helper.get_low_shot_output_file(opts, cls, cost, suffix)
    if not osp.exists(out_file):
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
        train_feats, train_cls_labels = svm_helper.get_cls_feats_labels(
            cls, features, targets, opts.dataset)
        clf.fit(train_feats, train_cls_labels)

        with open(out_file, 'wb') as fwrite:
            pickle.dump(clf, fwrite)
    return 0


def mp_helper(args):
    return task(*args)


def train_svm_low_shot(opts):
    """Train the svm low-shot model."""
    assert osp.exists(opts.data_file), 'Data file not found. Abort!'
    if not osp.exists(opts.output_path):
        os.makedirs(opts.output_path)

    features, targets = svm_helper.load_input_data(opts.data_file,
                                                   opts.targets_data_file)
    # normalize the features: N x 9216 (example shape)
    features = svm_helper.normalize_features(features)

    # parse the cost values for training the SVM on
    costs_list = svm_helper.parse_cost_list(opts.costs_list)

    # classes for which SVM testing should be done
    num_classes, cls_list = svm_helper.get_low_shot_svm_classes(
        targets, opts.dataset)

    num_task = len(cls_list) * len(costs_list)
    args_cls = []
    args_cost = []
    for cls in cls_list:
        for cost in costs_list:
            args_cls.append(cls)
            args_cost.append(cost)
    args_opts = [opts] * num_task
    args_features = [features] * num_task
    args_targets = [targets] * num_task

    pool = mp.Pool(mp.cpu_count())
    for _ in tqdm.tqdm(
            pool.imap_unordered(
                mp_helper,
                zip(args_cls, args_cost, args_opts, args_features,
                    args_targets)),
            total=num_task):
        pass


def main():
    parser = argparse.ArgumentParser(description='Low-shot SVM model training')
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
        '--costs_list',
        type=str,
        default='0.01,0.1',
        help='comma separated string containing list of costs')
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='path where to save the trained SVM models')
    parser.add_argument(
        '--random_seed',
        type=int,
        default=100,
        help='random seed for SVM classifier training')
    parser.add_argument(
        '--dataset', type=str, default='voc', help='voc | places')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()
    train_svm_low_shot(opts)


if __name__ == '__main__':
    main()
