#!/usr/bin/env bash

set -x

CFG=$1
CHECKPOINT=$2
GPUS=${GPUS:-8}
PY_ARGS=${@:3}


PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
mim test mmcls \
    $CFG \
    --checkpoint $CHECKPOINT \
    --launcher pytorch \
    -G $GPUS \
    --cfg-options $PY_ARGS
