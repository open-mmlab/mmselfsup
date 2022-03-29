#!/usr/bin/env bash

set -e
set -x

CFG=$1
EPOCH=$2
FEAT_LIST=$3  # e.g.: "feat5", "feat4 feat5". If leave empty, the default is "feat5"
GPUS=${GPUS:-8}
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

if [ "$CFG" == "" ] || [ "$EPOCH" == "" ]; then
    echo "ERROR: Missing arguments."
    exit
fi

if [ ! -f $WORK_DIR/epoch_${EPOCH}.pth ]; then
    echo "ERROR: File not exist: $WORK_DIR/epoch_${EPOCH}.pth"
    exit
fi

mkdir -p $WORK_DIR/logs
echo "Testing checkpoint: $WORK_DIR/epoch_${EPOCH}.pth" 2>&1 | tee -a $WORK_DIR/logs/eval_svm.log

bash tools/benchmarks/classification/svm_voc07/dist_extract.sh $CFG $GPUS $WORK_DIR --checkpoint $WORK_DIR/epoch_${EPOCH}.pth

bash tools/benchmarks/classification/svm_voc07/tools/eval_svm_full.sh $WORK_DIR "$FEAT_LIST"

bash tools/benchmarks/classification/svm_voc07/tools/eval_svm_lowshot.sh $WORK_DIR "$FEAT_LIST"
