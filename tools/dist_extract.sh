#!/usr/bin/env bash
PYTHON=${PYTHON:-"python"}
CFG=$1
CHECKPOINT=$2
GPUS=${3:-8}
PORT=${PORT:-29500}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
if [ "$CHECKPOINT" == "" ]; then
    $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        tools/extract.py $CFG --layer-ind "0,1,2,3,4" --work_dir $WORK_DIR --launcher pytorch
else
    $PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        tools/extract.py $CFG --layer-ind "0,1,2,3,4" --checkpoint $CHECKPOINT \
        --work_dir $WORK_DIR --launcher pytorch
fi
