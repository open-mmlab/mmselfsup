#!/usr/bin/env bash

set -x

CFG=$1
GPUS=$2
WORK_DIR=$3
PY_ARGS=${@:4}
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/benchmarks/classification/svm_voc07/extract.py $CFG \
    --layer-ind "0,1,2,3,4" --work_dir $WORK_DIR \
    --launcher pytorch ${PY_ARGS}
