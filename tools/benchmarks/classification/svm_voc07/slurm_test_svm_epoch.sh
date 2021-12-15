#!/usr/bin/env bash

set -e
set -x

PARTITION=$1
JOB_NAME=$2
CFG=$3
EPOCH=$4
FEAT_LIST=$5  # e.g.: "feat5", "feat4 feat5". If leave empty, the default is "feat5"
PY_ARGS=${@:6}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

if [ ! -f $WORK_DIR/epoch_${EPOCH}.pth ]; then
    echo "ERROR: File not exist: $WORK_DIR/epoch_${EPOCH}.pth"
    exit
fi

mkdir -p $WORK_DIR/logs
echo "Testing checkpoint: $WORK_DIR/epoch_${EPOCH}.pth" 2>&1 | tee -a $WORK_DIR/logs/eval_svm.log

bash tools/benchmarks/classification/svm_voc07/slurm_extract.sh $PARTITION $JOB_NAME $CFG $WORK_DIR --checkpoint $WORK_DIR/epoch_${EPOCH}.pth ${PY_ARGS}

srun -p $PARTITION --job-name=${JOB_NAME} ${SRUN_ARGS} bash tools/benchmarks/classification/svm_voc07/tools/eval_svm_full.sh $WORK_DIR "$FEAT_LIST"

srun -p $PARTITION --job-name=${JOB_NAME} ${SRUN_ARGS} bash tools/benchmarks/classification/svm_voc07/tools/eval_svm_lowshot.sh $WORK_DIR "$FEAT_LIST"
