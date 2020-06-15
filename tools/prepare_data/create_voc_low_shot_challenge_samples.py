# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################
"""
This script is used to create the low-shot data for VOC svm trainings.
"""
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import json
import logging
import numpy as np
import os
import random
import sys

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def load_json(file_path, ground_truth=True):
    import json
    assert os.path.exists(file_path), "{} does not exist".format(file_path)
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    img_ids = sorted(list(data.keys()))
    cls_names = sorted(list(data[img_ids[0]].keys()))
    if ground_truth:
        output = np.empty((len(img_ids), len(cls_names)), dtype=np.int32)
    else:
        output = np.empty((len(img_ids), len(cls_names)), dtype=np.float64)
    for idx in range(len(img_ids)):
        for cls_idx in range(len(cls_names)):
            output[idx][cls_idx] = data[img_ids[idx]][cls_names[cls_idx]]
    return output, img_ids, cls_names


def save_json(input_data, img_ids, cls_names, output_file):
    output_dict = {}
    for img_idx in range(len(img_ids)):
        img_id = img_ids[img_idx]
        out_lbl = {}
        for cls_idx in range(len(cls_names)):
            name = cls_names[cls_idx]
            out_lbl[name] = int(input_data[img_idx][cls_idx])
        output_dict[img_id] = out_lbl
    logger.info('Saving file: {}'.format(output_file))
    with open(output_file, 'w') as fp:
        json.dump(output_dict, fp)


def sample_symbol(input_targets, output_target, symbol, num):
    logger.info('Sampling symbol: {} for num: {}'.format(symbol, num))
    num_classes = input_targets.shape[1]
    for idx in range(num_classes):
        symbol_data = np.where(input_targets[:, idx] == symbol)[0]
        sampled = random.sample(list(symbol_data), num)
        for index in sampled:
            output_target[index, idx] = symbol
    return output_target


def generate_independent_sample(opts, targets, img_ids, cls_names):
    k_values = [int(val) for val in opts.k_values.split(",")]
    # the way sample works is: for each independent sample, and a given k value
    # we create a matrix of the same shape as given targets file. We initialize
    # this matrix with -1 (ignore label). We then sample k positive and
    # (num_classes-1) * k negatives.
    # N x 20 shape
    num_classes = targets.shape[1]
    for idx in range(opts.num_samples):
        for k in k_values:
            logger.info('Sampling: {} time for k-value: {}'.format(idx + 1, k))
            output = np.ones(targets.shape, dtype=np.int32) * -1
            output = sample_symbol(targets, output, 1, k)
            output = sample_symbol(targets, output, 0, (num_classes - 1) * k)
            prefix = opts.targets_data_file.split('/')[-1].split('.')[0]
            output_file = os.path.join(
                opts.output_path,
                '{}_sample{}_k{}.json'.format(prefix, idx + 1, k))
            save_json(output, img_ids, cls_names, output_file)
            npy_output_file = os.path.join(
                opts.output_path,
                '{}_sample{}_k{}.npy'.format(prefix, idx + 1, k))
            logger.info('Saving npy file: {}'.format(npy_output_file))
            np.save(npy_output_file, output)
    logger.info('Done!!')


def main():
    parser = argparse.ArgumentParser(
        description='Sample Low shot data for VOC')
    parser.add_argument(
        '--targets_data_file',
        type=str,
        default=None,
        help="Json file containing image labels")
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help="path where low-shot samples should be saved")
    parser.add_argument(
        '--k_values',
        type=str,
        default="1,2,4,8,16,32,64,96",
        help="Low-shot k-values for svm testing.")
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help="Number of independent samples.")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    opts = parser.parse_args()
    targets, img_ids, cls_names = load_json(opts.targets_data_file)
    generate_independent_sample(opts, targets, img_ids, cls_names)


if __name__ == '__main__':
    main()
