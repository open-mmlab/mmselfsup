#!/usr/bin/env bash
set -e
set -x

PARTITION=$1
CFG=$2
PRETRAIN=$3 # pretrained model or "random" (random init)
FEAT_LIST=$4 # e.g.: "feat5", "feat4 feat5". If leave empty, the default is "feat5"
GPUS=${5:-8}
WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$(echo $PRETRAIN | rev | cut -d/ -f 1 | rev)"

if [ ! -f $PRETRAIN ] and [ "$PRETRAIN" != "random" ]; then
    echo "ERROR: PRETRAIN should be a file or a string \"random\", got: $PRETRAIN"
    exit
fi

mkdir -p $WORK_DIR/logs
echo "Testing pretrain: $PRETRAIN" 2>&1 | tee -a $WORK_DIR/logs/eval_svm.log

bash tools/srun_extract.sh $PARTITION $CFG $GPUS $WORK_DIR --pretrained $PRETRAIN

srun -p $PARTITION bash benchmarks/svm_tools/eval_svm_full.sh $WORK_DIR "$FEAT_LIST"

srun -p $PARTITION bash benchmarks/svm_tools/eval_svm_lowshot.sh $WORK_DIR "$FEAT_LIST"
