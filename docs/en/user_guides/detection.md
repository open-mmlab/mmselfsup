# Detection

- [Detection](#detection)
  - [Train](#train)

Here, we prefer to use MMDetection to do the detection task. First, make sure you have installed [MIM](https://github.com/open-mmlab/mim), which is also a project of OpenMMLab.

```shell
pip install openmim
```

It is very easy to install the package.

Besides, please refer to MMDet for [installation](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) and [data preparation](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md)

## Train

After installation, you can run MMDet with simple command.

```shell
# distributed version
bash tools/benchmarks/mmdetection/mim_dist_train.sh ${CONFIG} ${PRETRAIN} ${GPUS}

# slurm version
bash tools/benchmarks/mmdetection/mim_slurm_train.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
```

Remarks:

- `CONFIG`: Use config files under `configs/benchmarks/mmdetection/` or write your own config files
- `PRETRAIN`: the pre-trained model file.

Or if you want to do detection task with [detectron2](https://github.com/facebookresearch/detectron2), we also provides some config files.
Please refer to [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) for installation and follow the [directory structure](https://github.com/facebookresearch/detectron2/tree/main/datasets) to prepare your datasets required by detectron2.

```shell
conda activate detectron2 # use detectron2 environment here, otherwise use open-mmlab environment
cd benchmarks/detection
python convert-pretrain-to-detectron2.py ${WEIGHT_FILE} ${OUTPUT_FILE} # must use .pkl as the output extension.
bash run.sh ${DET_CFG} ${OUTPUT_FILE}
```
