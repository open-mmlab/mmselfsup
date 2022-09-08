#!/usr/bin/env bash

set -x

PARTITION=$1
CFG=$2  # use cfgs under "configs/benchmarks/classification/imagenet/*.py"
CHECKPOINT=$3  # pretrained model
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:4}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
mim test mmcls $CFG \
    --checkpoint $CHECKPOINT \
    --launcher slurm -G $GPUS \
    --gpus-per-node $GPUS_PER_NODE \
    --cpus-per-task $CPUS_PER_TASK \
    --partition $PARTITION \
    --srun-args "$SRUN_ARGS" \
    --cfg-options $PY_ARGS \
