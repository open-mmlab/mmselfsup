# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
###############################################################################
"""Helper module for svm training and testing."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import json
import logging
import os.path as osp
import sys

import numpy as np

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def py2_py3_compatible_cost(cost):
    """Python 2 and python 3 have different floating point precision.

    The following trick helps keep the backwards compatibility.
    """
    return str(float(f'{cost:.17f}'))


def load_json(file_path):
    """Load json file."""
    assert osp.exists(file_path), f'{file_path} does not exist'
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    img_ids = list(data.keys())
    cls_names = list(data[img_ids[0]].keys())
    return img_ids, cls_names


def get_svm_train_output_files(cls, cost, output_path):
    """Get output file path."""
    cls_cost = str(cls) + '_cost' + py2_py3_compatible_cost(cost)
    out_file = osp.join(output_path, 'cls' + cls_cost + '.pickle')
    ap_matrix_out_file = osp.join(output_path, 'AP_cls' + cls_cost + '.npy')
    return out_file, ap_matrix_out_file


def parse_cost_list(costs):
    """Parse cost list."""
    costs_list = [float(cost) for cost in costs.split(',')]
    start_num, end_num = 4, 20
    for num in range(start_num, end_num):
        costs_list.append(0.5**num)
    return costs_list


def normalize_features(features):
    """normalization."""
    feats_norm = np.linalg.norm(features, axis=1)
    features = features / (feats_norm + 1e-5)[:, np.newaxis]
    return features


def load_input_data(data_file, targets_file):
    """Load the features and the targets."""
    targets = np.load(targets_file, encoding='latin1')
    features = np.array(np.load(data_file,
                                encoding='latin1')).astype(np.float64)
    assert features.shape[0] == targets.shape[0], 'Mismatched #images'
    return features, targets


def calculate_ap(rec, prec):
    """Computes the AP under the precision recall curve."""
    rec, prec = rec.reshape(rec.size, 1), prec.reshape(prec.size, 1)
    z, o = np.zeros((1, 1)), np.ones((1, 1))
    mrec, mpre = np.vstack((z, rec, o)), np.vstack((z, prec, z))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    indices = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = 0
    for i in indices:
        ap = ap + (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


def get_precision_recall(targets, preds):
    """
    [P, R, score, ap] = get_precision_recall(targets, preds)
    Input    :
        targets  : number of occurrences of this class in the ith image
        preds    : score for this image
    Output   :
        P, R   : precision and recall
        score  : score which corresponds to the particular precision and recall
        ap     : average precision
    """
    # binarize targets
    targets = np.array(targets > 0, dtype=np.float32)
    tog = np.hstack((targets[:, np.newaxis].astype(np.float64),
                     preds[:, np.newaxis].astype(np.float64)))
    ind = np.argsort(preds)
    ind = ind[::-1]
    score = np.array([tog[i, 1] for i in ind])
    sortcounts = np.array([tog[i, 0] for i in ind])

    tp = sortcounts
    fp = sortcounts.copy()
    for i in range(sortcounts.shape[0]):
        if sortcounts[i] >= 1:
            fp[i] = 0.
        elif sortcounts[i] < 1:
            fp[i] = 1.
    P = np.cumsum(tp) / (np.cumsum(tp) + np.cumsum(fp))
    numinst = np.sum(targets)
    R = np.cumsum(tp) / numinst
    ap = calculate_ap(R, P)
    return P, R, score, ap


def get_low_shot_output_file(opts, cls, cost, suffix):
    """in case of low-shot training, we train for 5 independent samples
    (sample{}) and vary low-shot amount (k{}).

    The input data should have sample{}_k{} information that we extract in
    suffix below.
    """
    cls_cost = str(cls) + '_cost' + py2_py3_compatible_cost(cost)
    out_file = osp.join(opts.output_path,
                        'cls' + cls_cost + '_' + suffix + '.pickle')
    return out_file


def get_low_shot_svm_classes(targets, dataset):
    """Get num_classes and cls_list information by dataset type."""
    # classes for which SVM testing should be done
    num_classes, cls_list = None, None
    if dataset == 'voc':
        num_classes = targets.shape[1]
        cls_list = range(num_classes)
    elif dataset == 'places':
        # each image in places has a target cls [0, .... ,204]
        num_classes = len(set(targets[:, 0].tolist()))
        cls_list = list(set(targets[:, 0].tolist()))
    else:
        logger.info('Dataset not recognized. Abort!')
    return num_classes, cls_list


def get_cls_feats_labels(cls, features, targets, dataset):
    """Get out_feats and out_cls_labels information by dataset type."""
    out_feats, out_cls_labels = None, None
    if dataset == 'voc':
        cls_labels = targets[:, cls].astype(dtype=np.int32, copy=True)
        # find the indices for positive/negative imgs. Remove the ignore label.
        out_data_inds = (targets[:, cls] != -1)
        out_feats = features[out_data_inds]
        out_cls_labels = cls_labels[out_data_inds]
        # label 0 = not present, set it to -1 as svm train target.
        # Make the svm train target labels as -1, 1.
        out_cls_labels[np.where(out_cls_labels == 0)] = -1
    elif dataset == 'places':
        out_feats = features
        out_cls_labels = targets.astype(dtype=np.int32, copy=True)
        # for the given class, get the relevant positive/negative images and
        # make the label 1, -1
        cls_inds = np.where(targets[:, 0] == cls)
        non_cls_inds = (targets[:, 0] != cls)
        out_cls_labels[non_cls_inds] = -1
        out_cls_labels[cls_inds] = 1
        # finally reshape into the format taken by sklearn svm package.
        out_cls_labels = out_cls_labels.reshape(-1)
    else:
        raise Exception('args.dataset not recognized')
    return out_feats, out_cls_labels
