#!/usr/bin/env bash

set -e
set -x

PARTITION=$1
CFG=$2
EPOCH=$3
DATASET=$4 # imagenet or places205
PY_ARGS=${@:5}
JOB_NAME="openselfsup"
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

WORK_DIR=$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/
CHECKPOINT=$WORK_DIR/epoch_${EPOCH}.pth
WORK_DIR_EVAL=$WORK_DIR/${DATASET}_at_epoch_${EPOCH}/

# extract backbone
if [ ! -f "${CHECKPOINT::(-4)}_extracted.pth" ]; then
    srun -p ${PARTITION} \
        python tools/extract_backbone_weights.py $CHECKPOINT \
        --save-path ${CHECKPOINT::(-4)}_extracted.pth
fi

# train
GLOG_vmodule=MemcachedClient=-1 \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/train.py \
        configs/linear_classifier/${DATASET}/r50_multihead.py \
        --pretrained ${CHECKPOINT::(-4)}_extracted.pth \
        --work_dir ${WORK_DIR_EVAL} --seed 0 --launcher="slurm" ${PY_ARGS}

# test
GLOG_vmodule=MemcachedClient=-1 \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/test.py \
        configs/linear_classifier/${DATASET}/r50_multihead.py \
        ${WORK_DIR_EVAL}/latest.pth \
        --work_dir ${WORK_DIR_EVAL} --launcher="slurm"
