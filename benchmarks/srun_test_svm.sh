#!/usr/bin/env bash
set -e
set -x

PARTITION=$1
CFG=$2
EPOCH=$3
FEAT=$4
WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

bash tools/srun_extract.sh $PARTITION $CFG $WORK_DIR/epoch_${EPOCH}.pth

srun -p $PARTITION bash benchmarks/eval_svm.sh $WORK_DIR $FEAT

srun -p $PARTITION bash benchmarks/eval_svm.sh $WORK_DIR $FEAT
