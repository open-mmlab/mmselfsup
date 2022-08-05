#!/usr/bin/env bash

set -x

CFG=$1
PRETRAIN=$2  # pretrained model
GPUS=$3
PY_ARGS=${@:4}

# set work_dir according to config path and pretrained model to distinguish different models
WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$(echo $PRETRAIN | rev | cut -d/ -f 1 | rev)"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
mim train mmseg $CFG \
    --launcher pytorch -G $GPUS \
    --work-dir $WORK_DIR \
    --cfg-options \
    model.backbone.pretrained=$PRETRAIN \
    $PY_ARGS
