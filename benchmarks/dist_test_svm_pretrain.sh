#!/bin/bash
set -e
set -x

CFG=$1
PRETRAIN=$2 # pretrained model or "random" (random init)
FEAT_LIST=$3 # e.g.: "feat5", "feat4 feat5". If leave empty, the default is "feat5"
GPUS=${4:-8}
WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$(echo $PRETRAIN | rev | cut -d/ -f 1 | rev)"

if [ "$CFG" == "" ] || [ "$PRETRAIN" == "" ]; then
    echo "ERROR: Missing arguments."
    exit
fi

if [ ! -f $PRETRAIN ] && [ "$PRETRAIN" != "random" ]; then
    echo "ERROR: PRETRAIN should be a file or a string \"random\", got: $PRETRAIN"
    exit
fi

mkdir -p $WORK_DIR/logs
echo "Testing pretrain: $PRETRAIN" 2>&1 | tee -a $WORK_DIR/logs/eval_svm.log

bash tools/dist_extract.sh $CFG $GPUS $WORK_DIR --pretrained $PRETRAIN

bash benchmarks/svm_tools/eval_svm_full.sh $WORK_DIR "$FEAT_LIST"

bash benchmarks/svm_tools/eval_svm_lowshot.sh $WORK_DIR "$FEAT_LIST"
