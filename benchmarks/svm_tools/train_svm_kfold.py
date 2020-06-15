# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################
"""
SVM training using 3-fold cross-validation.

Relevant transfer tasks: Image Classification VOC07 and COCO2014.
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
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

import svm_helper

import time

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def train_svm(opts):
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

    # classes for which SVM training should be done
    if opts.cls_list:
        cls_list = [int(cls) for cls in opts.cls_list.split(",")]
    else:
        num_classes = targets.shape[1]
        cls_list = range(num_classes)
    #logger.info('Training SVM for classes: {}'.format(cls_list))

    for cls_idx in tqdm(range(len(cls_list))):
        cls = cls_list[cls_idx]
        for cost_idx in range(len(costs_list)):
            start = time.time()
            cost = costs_list[cost_idx]
            out_file, ap_out_file = svm_helper.get_svm_train_output_files(
                cls, cost, opts.output_path)
            if os.path.exists(out_file) and os.path.exists(ap_out_file):
                logger.info('SVM model exists: {}'.format(out_file))
                logger.info('AP file exists: {}'.format(ap_out_file))
            else:
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
                cls_labels = targets[:, cls].astype(dtype=np.int32, copy=True)
                # meaning of labels in VOC/COCO original loaded target files:
                # label 0 = not present, set it to -1 as svm train target
                # label 1 = present. Make the svm train target labels as -1, 1.
                cls_labels[np.where(cls_labels == 0)] = -1
                #num_positives = len(np.where(cls_labels == 1)[0])
                #num_negatives = len(cls_labels) - num_positives

                #logger.info('cls: {} has +ve: {} -ve: {} ratio: {}'.format(
                #    cls, num_positives, num_negatives,
                #    float(num_positives) / num_negatives)
                #)
                #logger.info('features: {} cls_labels: {}'.format(
                #    features.shape, cls_labels.shape))
                ap_scores = cross_val_score(
                    clf,
                    features,
                    cls_labels,
                    cv=3,
                    scoring='average_precision')
                clf.fit(features, cls_labels)

                #logger.info('cls: {} cost: {} AP: {} mean:{}'.format(
                #    cls, cost, ap_scores, ap_scores.mean()))
                #logger.info('Saving cls cost AP to: {}'.format(ap_out_file))
                np.save(ap_out_file, np.array([ap_scores.mean()]))
                #logger.info('Saving SVM model to: {}'.format(out_file))
                with open(out_file, 'wb') as fwrite:
                    pickle.dump(clf, fwrite)
            print("time: {:.4g} s".format(time.time() - start))


def main():
    parser = argparse.ArgumentParser(description='SVM model training')
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
        '--output_path',
        type=str,
        default=None,
        help="path where to save the trained SVM models")
    parser.add_argument(
        '--costs_list',
        type=str,
        default="0.01,0.1",
        help="comma separated string containing list of costs")
    parser.add_argument(
        '--random_seed',
        type=int,
        default=100,
        help="random seed for SVM classifier training")

    parser.add_argument(
        '--cls_list',
        type=str,
        default=None,
        help="comma separated string list of classes to train")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()
    #logger.info(opts)
    train_svm(opts)


if __name__ == '__main__':
    main()
