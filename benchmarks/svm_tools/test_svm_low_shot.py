# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
################################################################################
"""
SVM test for low shot image classification.

Relevant transfer tasks: Low-shot Image Classification VOC07 and Places205 low
shot samples.
"""
from __future__ import division
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import argparse
import json
import logging
import numpy as np
import os
import pickle
import six
import sys

import svm_helper

# create the logger
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def load_json(file_path):
    assert os.path.exists(file_path), "{} does not exist".format(file_path)
    with open(file_path, 'r') as fp:
        data = json.load(fp)
    img_ids = list(data.keys())
    cls_names = list(data[img_ids[0]].keys())
    return img_ids, cls_names


def save_json_predictions(opts, cost, sample_idx, k_low, features, cls_list,
                          cls_names, img_ids):
    num_classes = len(cls_list)
    json_predictions = {}
    for cls in range(num_classes):
        suffix = 'sample{}_k{}'.format(sample_idx + 1, k_low)
        model_file = svm_helper.get_low_shot_output_file(
            opts, cls, cost, suffix)
        with open(model_file, 'rb') as fopen:
            if six.PY2:
                model = pickle.load(fopen)
            else:
                model = pickle.load(fopen, encoding='latin1')
        prediction = model.decision_function(features)
        cls_name = cls_names[cls]
        for idx in range(len(prediction)):
            img_id = img_ids[idx]
            if img_id in json_predictions:
                json_predictions[img_id][cls_name] = prediction[idx]
            else:
                out_lbl = {}
                out_lbl[cls_name] = prediction[idx]
                json_predictions[img_id] = out_lbl

    output_file = os.path.join(opts.output_path,
                               'test_{}_json_preds.json'.format(suffix))
    with open(output_file, 'w') as fp:
        json.dump(json_predictions, fp)
    #logger.info('Saved json predictions to: {}'.format(output_file))


def test_svm_low_shot(opts):
    k_values = [int(val) for val in opts.k_values.split(",")]
    sample_inds = [int(val) for val in opts.sample_inds.split(",")]
    #logger.info('Testing svm for k-values: {} and sample_inds: {}'.format(
    #    k_values, sample_inds))

    img_ids, cls_names = [], []
    if opts.generate_json:
        img_ids, cls_names = load_json(opts.json_targets)

    assert os.path.exists(opts.data_file), "Data file not found. Abort!"
    # we test the svms on the full test set. Given the test features and the
    # targets, we test it for various k-values (low-shot), cost values and
    # 5 independent samples.
    features, targets = svm_helper.load_input_data(opts.data_file,
                                                   opts.targets_data_file)
    # normalize the features: N x 9216 (example shape)
    features = svm_helper.normalize_features(features)

    # parse the cost values for training the SVM on
    costs_list = svm_helper.parse_cost_list(opts.costs_list)
    #logger.info('Testing SVM for costs: {}'.format(costs_list))

    # classes for which SVM testing should be done
    num_classes, cls_list = svm_helper.get_low_shot_svm_classes(
        targets, opts.dataset)

    # create the output for per sample, per k-value and per cost.
    sample_ap_matrices = []
    for _ in range(len(sample_inds)):
        ap_matrix = np.zeros((len(k_values), len(costs_list)))
        sample_ap_matrices.append(ap_matrix)

    # the test goes like this: For a given sample, for a given k-value and a
    # given cost value, we evaluate the trained svm model for all classes.
    # After computing over all classes, we get the mean AP value over all
    # classes. We hence end up with: output = [sample][k_value][cost]
    for inds in range(len(sample_inds)):
        sample_idx = sample_inds[inds]
        for k_idx in range(len(k_values)):
            k_low = k_values[k_idx]
            suffix = 'sample{}_k{}'.format(sample_idx + 1, k_low)
            for cost_idx in range(len(costs_list)):
                cost = costs_list[cost_idx]
                local_cost_ap = np.zeros((num_classes, 1))
                for cls in cls_list:
                    #logger.info(
                    #    'Test sample/k_value/cost/cls: {}/{}/{}/{}'.format(
                    #        sample_idx + 1, k_low, cost, cls))
                    model_file = svm_helper.get_low_shot_output_file(
                        opts, cls, cost, suffix)
                    with open(model_file, 'rb') as fopen:
                        if six.PY2:
                            model = pickle.load(fopen)
                        else:
                            model = pickle.load(fopen, encoding='latin1')
                    prediction = model.decision_function(features)
                    eval_preds, eval_cls_labels = svm_helper.get_cls_feats_labels(
                        cls, prediction, targets, opts.dataset)
                    P, R, score, ap = svm_helper.get_precision_recall(
                        eval_cls_labels, eval_preds)
                    local_cost_ap[cls][0] = ap
                mean_cost_ap = np.mean(local_cost_ap, axis=0)
                sample_ap_matrices[inds][k_idx][cost_idx] = mean_cost_ap
            out_k_sample_file = os.path.join(
                opts.output_path,
                'test_ap_sample{}_k{}.npy'.format(sample_idx + 1, k_low))
            save_data = sample_ap_matrices[inds][k_idx]
            save_data = save_data.reshape((1, -1))
            np.save(out_k_sample_file, save_data)
            #logger.info('Saved sample test k_idx AP to file: {} {}'.format(
            #    out_k_sample_file, save_data.shape))
            if opts.generate_json:
                argmax_cls = np.argmax(save_data, axis=1)
                chosen_cost = costs_list[argmax_cls[0]]
                #logger.info('chosen cost: {}'.format(chosen_cost))
                save_json_predictions(opts, chosen_cost, sample_idx, k_low,
                                      features, cls_list, cls_names, img_ids)
    #logger.info('All done!!')


def main():
    parser = argparse.ArgumentParser(description='Low shot SVM model test')
    parser.add_argument(
        '--data_file',
        type=str,
        default=None,
        help="Numpy file containing image features and labels")
    parser.add_argument(
        '--targets_data_file',
        type=str,
        default=None,
        help="Numpy file containing image labels")
    parser.add_argument(
        '--json_targets',
        type=str,
        default=None,
        help="Numpy file containing json targets")
    parser.add_argument(
        '--generate_json',
        type=int,
        default=0,
        help="Whether to generate json files for output")
    parser.add_argument(
        '--costs_list',
        type=str,
        default=
        "0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.1,1.0,10.0,100.0",
        help="comma separated string containing list of costs")
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help="path where trained SVM models are saved")
    parser.add_argument(
        '--k_values',
        type=str,
        default="1,2,4,8,16,32,64,96",
        help="Low-shot k-values for svm testing. Comma separated")
    parser.add_argument(
        '--sample_inds',
        type=str,
        default="0,1,2,3,4",
        help="sample_inds for which to test svm. Comma separated")
    parser.add_argument(
        '--dataset', type=str, default="voc", help='voc | places')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    opts = parser.parse_args()
    #logger.info(opts)
    test_svm_low_shot(opts)


if __name__ == '__main__':
    main()
