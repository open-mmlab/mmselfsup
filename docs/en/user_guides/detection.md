# 监测

- [监测](#监测)
  - [训练](#训练)
  - [测试](#测试)

在此，我们倾向于用 MMDection 完成监测任务。首先请确保您安装了OpenMMLab的 [MIM](https://github.com/open-mmlab/mim)　项目。

```shell
pip install openmim
mim install 'mmdet>=3.0.0rc0'
```

安装十分简单。

此外，请参考 MMDet　的[installation](https://mmdetection.readthedocs.io/en/dev-3.x/get_started.html) 和 [data preparation](https://mmdetection.readthedocs.io/en/dev-3.x/user_guides/dataset_prepare.html)

## 训练

安装后您可以用下面的简单命令运行 MMDection 。

```shell
# distributed 版本
bash tools/benchmarks/mmdetection/mim_dist_train_c4.sh ${CONFIG} ${PRETRAIN} ${GPUS}
bash tools/benchmarks/mmdetection/mim_dist_train_fpn.sh ${CONFIG} ${PRETRAIN} ${GPUS}

# slurm 版本
bash tools/benchmarks/mmdetection/mim_slurm_train_c4.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
bash tools/benchmarks/mmdetection/mim_slurm_train_fpn.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
```

注意:

- `CONFIG`: 用到 `configs/benchmarks/mmdetection/`　下的配置文件。因为　OpenMMLab 已支持跨仓库用配置文件，我们可以像下面这样简单的从　MMDection　中借用配置：

```shell
_base_ = 'mmdet::mask_rcnn/mask-rcnn_r50-caffe-c4_1x_coco.py'
```
我们也支持从零开始写您的配置文件。

- `PRETRAIN`: 预训练模型文件。
- `GPUS`: 您想用于训练的 GPU 数量。语义分割任务中默认用４块 GPU。

例如：

```shell
bash ./tools/benchmarks/mmdetection/mim_dist_train_c4.sh \
configs/benchmarks/mmdetection/coco/mask-rcnn_r50-c4_ms-1x_coco.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 8
```

或者如果您像用 [detectron2](https://github.com/facebookresearch/detectron2) 做监测任务, 我们也提供一些配置文件。安装部分请参考 [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) ，准备 detectron2 需要的数据集请参考 [directory structure](https://github.com/facebookresearch/detectron2/tree/main/datasets)。

```shell
conda activate detectron2 # use detectron2 environment here, otherwise use open-mmlab environment
cd tools/benchmarks/detectron2
python convert-pretrain-to-detectron2.py ${WEIGHT_FILE} ${OUTPUT_FILE} # must use .pkl as the output extension.
bash run.sh ${DET_CFG} ${OUTPUT_FILE}
```

## 测试

训练之后，您也可以用如下命令测试您的模型。

```shell
# distributed 版本
bash tools/benchmarks/mmdetection/mim_dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS}

# slurm 版本
bash tools/benchmarks/mmdetection/mim_slurm_test.sh ${PARTITION} ${CONFIG} ${CHECKPOINT}
```

注意:

- `CHECKPOINT`: 您想测试的训练好的模型。

例如：

```shell
bash ./tools/benchmarks/mmdetection/mim_dist_test.sh \
configs/benchmarks/mmdetection/coco/mask-rcnn_r50_fpn_ms-1x_coco.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 8
```
