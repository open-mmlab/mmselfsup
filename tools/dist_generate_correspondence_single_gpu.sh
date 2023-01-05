#!/bin/bash

set -x

CFG=$1
CHECKPOINT=$2
INPUT=$3
OUTPUT=$4
GPUS=1
PORT=${PORT:-29500}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

# test
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/generate_correspondence.py \
    $CFG \
    $CHECKPOINT \
    $INPUT \
    $OUTPUT \
    --work_dir $WORK_DIR --launcher="pytorch"
