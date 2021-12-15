#!/usr/bin/env bash

set -e
set -x

CFG=$1
GPUS=$2
PRETRAIN=$3  # pretrained model
FEAT_LIST=$4  # e.g.: "feat5", "feat4 feat5". If leave empty, the default is "feat5"
PY_ARGS=${@:5}

# set work_dir according to config path and pretrained model to distinguish different models
WORK_DIR="$(echo ${CFG%.*} | sed -e "s/configs/work_dirs/g")/$(echo $PRETRAIN | rev | cut -d/ -f 1 | rev)"

mkdir -p $WORK_DIR/logs
echo "Testing pretrain: $PRETRAIN" 2>&1 | tee -a $WORK_DIR/logs/eval_svm.log

bash tools/benchmarks/classification/svm_voc07/dist_extract.sh \
    $CFG $GPUS $WORK_DIR \
    --cfg-options model.backbone.init_cfg.type=Pretrained model.backbone.init_cfg.checkpoint=$PRETRAIN \
    ${PY_ARGS}

bash tools/benchmarks/classification/svm_voc07/tools/eval_svm_full.sh $WORK_DIR "$FEAT_LIST"

bash tools/benchmarks/classification/svm_voc07/tools/eval_svm_lowshot.sh $WORK_DIR "$FEAT_LIST"
