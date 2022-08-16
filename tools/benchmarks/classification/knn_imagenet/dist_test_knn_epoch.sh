#!/usr/bin/env bash

set -e
set -x

CFG=$1
EPOCH=$2
PY_ARGS=${@:3}
GPUS=${GPUS:-8}
PORT=${PORT:-29500}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

if [ ! -f $WORK_DIR/epoch_${EPOCH}.pth ]; then
    echo "ERROR: File not exist: $WORK_DIR/epoch_${EPOCH}.pth"
    exit
fi

python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/benchmarks/classification/knn_imagenet/test_knn.py $CFG \
    --checkpoint $WORK_DIR/epoch_${EPOCH}.pth \
    --work-dir $WORK_DIR --launcher="pytorch" ${PY_ARGS}
