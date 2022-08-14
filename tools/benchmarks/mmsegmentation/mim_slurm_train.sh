#!/usr/bin/env bash

set -x

PARTITION=$1
CFG=$2
PRETRAIN=$3  # pretrained model
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}

# set work_dir according to config path and pretrained model to distinguish different models
WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$(echo $PRETRAIN | rev | cut -d/ -f 1 | rev)"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
mim train mmseg $CFG \
    --launcher slurm -G $GPUS \
    --gpus-per-node $GPUS_PER_NODE \
    --cpus-per-task $CPUS_PER_TASK \
    --partition $PARTITION \
    --work-dir $WORK_DIR \
    --srun-args "$SRUN_ARGS" \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=$PRETRAIN \
    model.backbone.init_cfg.prefix="backbone." \
    $PY_ARGS
