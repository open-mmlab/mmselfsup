#!/bin/bash

set -x
set -e

WORK_DIR=$1
MODE="full"
FEAT_LIST=${2:-"feat5"} # "feat1 feat2 feat3 feat4 feat5"
TRAIN_SVM_LOWSHOT_FLAG=true
TEST_SVM_LOWSHOT_FLAG=true
AGGREGATE_FLAG=true
DATA="data/VOCdevkit/VOC2007/SVMLabels"

# config svm
costs="1.0,10.0,100.0"
if [ "$MODE" == "fast" ]; then
    shots="96"
else
    shots="1 2 4 8 16 32 64 96"
fi

for feat in $FEAT_LIST; do
    echo "For feature: $feat" 2>&1 | tee -a $WORK_DIR/logs/eval_svm.log
    # train lowshot svm
    if $TRAIN_SVM_LOWSHOT_FLAG; then
        rm -rf $WORK_DIR/svm_lowshot
        mkdir -p $WORK_DIR/svm_lowshot/voc07_${feat}
        echo "training svm low-shot ..."
        for s in {1..5}; do
            for k in $shots; do
                echo -e "\ts${s} k${k}"
                python tools/benchmarks/classification/svm_voc07/tools/train_svm_low_shot.py \
                    --data_file $WORK_DIR/features/voc07_trainval_${feat}.npy \
                    --targets_data_file $DATA/low_shot/labels/train_targets_sample${s}_k${k}.npy \
                    --costs_list $costs \
                    --output_path $WORK_DIR/svm_lowshot/voc07_${feat}
            done
        done
    fi

    # test lowshot svm
    if $TEST_SVM_LOWSHOT_FLAG; then
        echo "testing svm low-shot ..."
        python tools/benchmarks/classification/svm_voc07/tools/test_svm_low_shot.py \
            --data_file $WORK_DIR/features/voc07_test_${feat}.npy \
            --targets_data_file $DATA/test_labels.npy \
            --json_targets $DATA/test_targets.json \
            --generate_json 1 \
            --costs_list $costs \
            --output_path $WORK_DIR/svm_lowshot/voc07_${feat} \
            --k_values "${shots// /,}" \
            --sample_inds "0,1,2,3,4" \
            --dataset "voc"
    fi

    # aggregate testing results
    if $AGGREGATE_FLAG; then
        echo "aggregating svm low-shot ..."
        python tools/benchmarks/classification/svm_voc07/tools/aggregate_low_shot_svm_stats.py \
            --output_path $WORK_DIR/svm_lowshot/voc07_${feat} \
            --k_values "${shots// /,}" \
            --sample_inds "0,1,2,3,4" 2>&1 | tee -a $WORK_DIR/logs/eval_svm.log
    fi

done
