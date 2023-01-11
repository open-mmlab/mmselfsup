#!/usr/bin/env bash

set -x
PARTITION=$1
JOB_NAME='correspondence'
CFG=$2
CHECKPOINT=$3
INPUT=$4
OUTPUT=$5
PY_ARGS=${@:6}
GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u tools/generate_correspondence.py \
        $CFG \
        $CHECKPOINT \
        $INPUT \
        $OUTPUT --launcher="slurm" ${PY_ARGS}
