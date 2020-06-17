#!/usr/bin/env bash

set -e
set -x

CFG=$1 # use cfgs under "configs/classification/imagenet_*percent/"
PRETRAIN=$2
PY_ARGS=${@:3}
GPUS=8 # in the standard setting, GPUS=8

if [ "$CFG" == "" ] || [ "$PRETRAIN" == "" ]; then
    echo "ERROR: Missing arguments."
    exit
fi

WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$(echo $PRETRAIN | rev | cut -d/ -f 1 | rev)"

# train
python -m torch.distributed.launch --nproc_per_node=$GPUS \
    tools/train.py \
    $CFG \
    --pretrained $PRETRAIN \
    --work_dir $WORK_DIR --seed 0 --launcher="pytorch" ${PY_ARGS}

# test
python -m torch.distributed.launch --nproc_per_node=$GPUS \
    tools/test.py \
    $CFG \
    $WORK_DIR/latest.pth \
    --work_dir $WORK_DIR --launcher="pytorch"
