#!/usr/bin/env bash

set -e
set -x

CFG=$1
EPOCH=$2
DATASET=$3 # imagenet or places205
GPUS=${GPUS:-1}
PORT=${PORT:-29500}
PY_ARGS=${@:4}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
CHECKPOINT=$WORK_DIR/epoch_${EPOCH}.pth
WORK_DIR_EVAL=$WORK_DIR/${DATASET}_at_epoch_${EPOCH}/

# extract backbone
if [ ! -f "${CHECKPOINT::(-4)}_extracted.pth" ]; then
    python tools/extract_backbone_weights.py $CHECKPOINT \
        --save-path ${CHECKPOINT::(-4)}_extracted.pth
fi

# train
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py \
    configs/linear_classification/${DATASET}/r50_multihead.py \
    --pretrained ${CHECKPOINT::(-4)}_extracted.pth \
    --work_dir ${WORK_DIR_EVAL} --seed 0 --launcher="pytorch" ${PY_ARGS}

# test
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py \
    configs/linear_classification/${DATASET}/r50_multihead.py \
    ${WORK_DIR_EVAL}/latest.pth \
    --work_dir ${WORK_DIR_EVAL} --launcher="pytorch"
