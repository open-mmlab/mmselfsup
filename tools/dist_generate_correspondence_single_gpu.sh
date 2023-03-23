#!/usr/bin/env bash

CFG=$1
CHECKPOINT=$2
INPUT=$3
OUTPUT=$4
PY_ARGS=${@:5}
GPUS=1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/generate_correspondence.py \
    $CFG \
    $CHECKPOINT \
    $INPUT \
    $OUTPUT --launcher pytorch ${PY_ARGS}
