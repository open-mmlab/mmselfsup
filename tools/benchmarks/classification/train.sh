#!/usr/bin/env bash

set -x

PARTITION=$1
CFG=$2  # use cfgs under "configs/benchmarks/classification/imagenet/*.py"
PRETRAIN=$3  # pretrained model
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --quotatype=auto \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python -u tools/train.py ${CONFIG} \
    --launcher="slurm" \
    --work-dir $WORK_DIR \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=${PRETRAIN} \
    model.backbone.init_cfg.prefix="backbone." \
    ${PY_ARGS}