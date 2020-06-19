#!/usr/bin/env bash
set -x

PARTITION=$1
CFG=$2
GPUS=$3
WORK_DIR=$4
PY_ARGS=${@:5} # "--checkpoint $CHECKPOINT --pretrained $PRETRAINED"
JOB_NAME="openselfsup"
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/extract.py $CFG \
        --layer-ind "0,1,2,3,4" --work_dir $WORK_DIR \
        --launcher="slurm" ${PY_ARGS}
