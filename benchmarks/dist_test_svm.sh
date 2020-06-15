#!/bin/bash
set -e
set -x

CFG=$1
EPOCH=$2
FEAT_LIST=$3
GPUS=$4
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

bash tools/dist_extract.sh $CFG $WORK_DIR/epoch_${EPOCH}.pth $GPUS

bash benchmarks/eval_svm.sh $WORK_DIR $FEAT_LIST

bash benchmarks/eval_svm_lowshot.sh $WORK_DIR $FEAT_LIST
