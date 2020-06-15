# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################
"""
Low Shot SVM training.

Relevant transfer tasks: Low-shot Image Classification VOC07 and Places205 low
shot samples.
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import argparse
import logging
import numpy as np
import os
import pickle
import sys
from sklearn.svm import LinearSVC
from tqdm import tqdm

import svm_helper

import time

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def train_svm_low_shot(opts):
    assert os.path.exists(opts.data_file), "Data file not found. Abort!"
    if not os.path.exists(opts.output_path):
        os.makedirs(opts.output_path)

    features, targets = svm_helper.load_input_data(opts.data_file,
                                                   opts.targets_data_file)
    # normalize the features: N x 9216 (example shape)
    features = svm_helper.normalize_features(features)

    # parse the cost values for training the SVM on
    costs_list = svm_helper.parse_cost_list(opts.costs_list)
    #logger.info('Training SVM for costs: {}'.format(costs_list))

    # classes for which SVM testing should be done
    num_classes, cls_list = svm_helper.get_low_shot_svm_classes(
        targets, opts.dataset)

    for cls in tqdm(cls_list):
        for cost_idx in range(len(costs_list)):
            start = time.time()
            cost = costs_list[cost_idx]
            suffix = '_'.join(
                opts.targets_data_file.split('/')[-1].split('.')[0].split('_')
                [-2:])
            out_file = svm_helper.get_low_shot_output_file(
                opts, cls, cost, suffix)
            if os.path.exists(out_file):
                logger.info('SVM model exists: {}'.format(out_file))
            else:
                #logger.info('SVM model not found: {}'.format(out_file))
                #logger.info('Training model with the cost: {}'.format(cost))
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
                #num_positives = len(np.where(train_cls_labels == 1)[0])
                #num_negatives = len(np.where(train_cls_labels == -1)[0])

                #logger.info('cls: {} has +ve: {} -ve: {} ratio: {}'.format(
                #    cls, num_positives, num_negatives,
                #    float(num_positives) / num_negatives)
                #)
                #logger.info('features: {} cls_labels: {}'.format(
                #    train_feats.shape, train_cls_labels.shape))
                clf.fit(train_feats, train_cls_labels)
                #logger.info('Saving SVM model to: {}'.format(out_file))
                with open(out_file, 'wb') as fwrite:
                    pickle.dump(clf, fwrite)
            #print("time: {:.4g} s".format(time.time() - start))
    #logger.info('All done!')


def main():
    parser = argparse.ArgumentParser(description='Low-shot SVM model training')
    parser.add_argument(
        '--data_file',
        type=str,
        default=None,
        help="Numpy file containing image features")
    parser.add_argument(
        '--targets_data_file',
        type=str,
        default=None,
        help="Numpy file containing image labels")
    parser.add_argument(
        '--costs_list',
        type=str,
        default="0.01,0.1",
        help="comma separated string containing list of costs")
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help="path where to save the trained SVM models")
    parser.add_argument(
        '--random_seed',
        type=int,
        default=100,
        help="random seed for SVM classifier training")
    parser.add_argument(
        '--dataset', type=str, default="voc", help='voc | places')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()

    #logger.info(opts)
    train_svm_low_shot(opts)


if __name__ == '__main__':
    main()
