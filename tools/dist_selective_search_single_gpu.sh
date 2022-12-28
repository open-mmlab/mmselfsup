#!/bin/bash

set -x

CFG=$1
OUTPUT=$2
GPUS=1
PORT=${PORT:-29500}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

# test
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/selective_search.py \
    $CFG \
    $OUTPUT \
    --work_dir $WORK_DIR --launcher="pytorch"
