#!/bin/bash

DATA="$1"
if [ "$DATA" == "" ]; then
    echo "Usage: bash tools/dataset_converters/prepare_voc07_cls.sh YOUR_DATA_ROOT"
    exit
fi

VOC="$DATA/VOCdevkit/VOC2007/"

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar -P $DATA
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar -P $DATA
tar -xf $DATA/VOCtrainval_06-Nov-2007.tar -C $DATA
tar -xf $DATA/VOCtest_06-Nov-2007.tar -C $DATA

mkdir -p $VOC/SVMLabels/low_shot/labels/

python $(dirname "$0")/create_voc_data_files.py \
    --data_source_dir $VOC \
    --output_dir $VOC/SVMLabels/ \
    --generate_json 1

python $(dirname "$0")/create_voc_low_shot_challenge_samples.py \
    --targets_data_file $VOC/SVMLabels/train_targets.json \
    --output_path $VOC/SVMLabels/low_shot/labels/ \
    --k_values "1,2,4,8,16,32,64,96" \
    --num_samples 5

mkdir $VOC/Lists

awk 'NF{print $0 ".jpg"}' $VOC/ImageSets/Main/trainval.txt $VOC/ImageSets/Main/test.txt > $VOC/Lists/trainvaltest.txt

mkdir -p data/
ln -s $DATA/VOCdevkit data/
