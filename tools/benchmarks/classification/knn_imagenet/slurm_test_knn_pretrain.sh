#!/usr/bin/env bash

set -e
set -x

PARTITION=$1
JOB_NAME=$2
CFG=$3
PRETRAIN=$4  # pretrained model
PY_ARGS=${@:5}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

# set work_dir according to config path and pretrained model to distinguish different models
WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$(echo $PRETRAIN | rev | cut -d/ -f 1 | rev)"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/benchmarks/classification/knn_imagenet/test_knn.py $CFG \
        --cfg-options model.backbone.init_cfg.type=Pretrained \
        model.backbone.init_cfg.checkpoint=$PRETRAIN \
        --work-dir $WORK_DIR --launcher="slurm" ${PY_ARGS}
