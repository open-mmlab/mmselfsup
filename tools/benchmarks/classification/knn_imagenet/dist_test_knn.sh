#!/usr/bin/env bash

set -e
set -x

CFG=$1
PRETRAIN=$2  # pretrained model
PY_ARGS=${@:3}
GPUS=${GPUS:-8}
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/benchmarks/classification/knn_imagenet/test_knn.py $CFG \
    --launcher="pytorch" \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=$PRETRAIN \
    model.backbone.init_cfg.prefix='backbone.' \
    ${PY_ARGS}
