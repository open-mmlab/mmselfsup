#!/usr/bin/env bash

set -e
set -x

CFG=$1
EPOCH=$2
PERCENT=$3
PY_ARGS=${@:4}
GPUS=${GPUS:-8}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
CHECKPOINT=$WORK_DIR/epoch_${EPOCH}.pth
WORK_DIR_EVAL=$WORK_DIR/imagenet_semi_${PERCENT}percent_at_epoch_${EPOCH}/

if [ ! "$PERCENT" == "1" ] && [ ! "$PERCENT" == 10 ]; then
    echo "ERROR: PERCENT must in {1, 10}"
    exit
fi

# extract backbone
if [ ! -f "${CHECKPOINT::(-4)}_extracted.pth" ]; then
    python tools/extract_backbone_weights.py $CHECKPOINT \
        --save-path ${CHECKPOINT::(-4)}_extracted.pth
fi

# train
python -m torch.distributed.launch --nproc_per_node=$GPUS \
    tools/train.py \
    configs/classification/imagenet_${PERCENT}percent/r50.py \
    --pretrained ${CHECKPOINT::(-4)}_extracted.pth \
    --work_dir ${WORK_DIR_EVAL} --seed 0 --launcher="pytorch" ${PY_ARGS}

# test
python -m torch.distributed.launch --nproc_per_node=$GPUS \
    tools/test.py \
    configs/classification/imagenet_${PERCENT}percent/r50.py \
    ${WORK_DIR_EVAL}/latest.pth \
    --work_dir ${WORK_DIR_EVAL} --launcher="pytorch"
