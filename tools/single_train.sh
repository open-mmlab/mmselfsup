#!/bin/bash
PYTHON=${PYTHON:-"python"}

CFG=$1
PY_ARGS=${@:2}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/

$PYTHON -u tools/train.py $1 --work_dir $WORK_DIR ${PY_ARGS}
