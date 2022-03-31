#!/usr/bin/env bash

set -e
set -x

CFG=$1  # use cfgs under "configs/benchmarks/classification/imagenet/*.py"
PRETRAIN=$2  # pretrained model
PY_ARGS=${@:3}
GPUS=${GPUS:-8}  # When changing GPUS, please also change samples_per_gpu in the config file accordingly to ensure the total batch size is 256.
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# set work_dir according to config path and pretrained model to distinguish different models
WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$(echo $PRETRAIN | rev | cut -d/ -f 1 | rev)"

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    tools/train.py $CFG \
    --cfg-options model.backbone.init_cfg.type=Pretrained \
    model.backbone.init_cfg.checkpoint=$PRETRAIN \
    --work-dir $WORK_DIR \
    --seed 0 \
    --launcher="pytorch" \
    ${PY_ARGS}
