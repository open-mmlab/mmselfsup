#!/bin/bash

set -x
set -e

WORK_DIR=$1
FEAT_LIST=${2:-"feat5"} # "feat1 feat2 feat3 feat4 feat5"
TRAIN_SVM_FLAG=true
TEST_SVM_FLAG=true
DATA="data/VOCdevkit/VOC2007/SVMLabels"

# config svm
costs="1.0,10.0,100.0"

for feat in $FEAT_LIST; do
    echo "For feature: $feat" 2>&1 | tee -a $WORK_DIR/logs/eval_svm.log
    # train svm
    if $TRAIN_SVM_FLAG; then
        rm -rf $WORK_DIR/svm
        mkdir -p $WORK_DIR/svm/voc07_${feat}
        echo "training svm ..."
        python tools/benchmarks/classification/svm_voc07/tools/train_svm_kfold_parallel.py \
            --data_file $WORK_DIR/features/voc07_trainval_${feat}.npy \
            --targets_data_file $DATA/train_labels.npy \
            --costs_list $costs \
            --output_path $WORK_DIR/svm/voc07_${feat}
    fi

    # test svm
    if $TEST_SVM_FLAG; then
        echo "testing svm ..."
        python tools/benchmarks/classification/svm_voc07/tools/test_svm.py \
            --data_file $WORK_DIR/features/voc07_test_${feat}.npy \
            --json_targets $DATA/test_targets.json \
            --targets_data_file $DATA/test_labels.npy \
            --costs_list $costs \
            --generate_json 1 \
            --output_path $WORK_DIR/svm/voc07_${feat} 2>&1 | tee -a $WORK_DIR/logs/eval_svm.log
    fi

done
