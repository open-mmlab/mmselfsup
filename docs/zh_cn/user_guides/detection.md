# 检测

- [检测](#detection)
  - [训练](#train)
  - [测试](#test)

这里，我们更喜欢使用MMDetection做检测任务。首先确保你已经安装了[MIM](https://github.com/open-mmlab/mim)，这也是OpenMMLab的一个项目。

```shell
pip install openmim
mim install 'mmdet>=3.0.0rc0'
```

非常容易安装这个包。

此外，请参考MMDet的[安装](https://mmdetection.readthedocs.io/en/dev-3.x/get_started.html)和[数据准备](https://mmdetection.readthedocs.io/en/dev-3.x/user_guides/dataset_prepare.html)

## 训练

安装完后，你可以使用如下的简单命令运行MMDetection。

```shell
# distributed version
bash tools/benchmarks/mmdetection/mim_dist_train_c4.sh ${CONFIG} ${PRETRAIN} ${GPUS}
bash tools/benchmarks/mmdetection/mim_dist_train_fpn.sh ${CONFIG} ${PRETRAIN} ${GPUS}

# slurm version
bash tools/benchmarks/mmdetection/mim_slurm_train_c4.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
bash tools/benchmarks/mmdetection/mim_slurm_train_fpn.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
```

注意：

- `CONFIG`: 使用`configs/benchmarks/mmdetection/`下的配置文件。由于OpenMMLab的存储库支持跨不同存储库引用配置文件，因此我们可以轻松使用MMDetection的配置文件，例如：

```shell
_base_ = 'mmdet::mask_rcnn/mask-rcnn_r50-caffe-c4_1x_coco.py'
```

从头开始写你的配置文件也是支持的。

- `PRETRAIN`：预训练模型文件
- `GPUS`: 你想用于训练的GPU数量，对于检测任务，我们默认采用8块GPU。

例子：

```shell
bash ./tools/benchmarks/mmdetection/mim_dist_train_c4.sh \
configs/benchmarks/mmdetection/coco/mask-rcnn_r50-c4_ms-1x_coco.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 8
```

或者你想用[detectron2](https://github.com/facebookresearch/detectron2)来做检测任务，我们也提供了一些配置文件。
请参考[INSTALL.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)用于安装并按照detectron2需要的[目录结构](https://github.com/facebookresearch/detectron2/tree/main/datasets)准备你的数据集。

```shell
conda activate detectron2 # use detectron2 environment here, otherwise use open-mmlab environment
cd tools/benchmarks/detectron2
python convert-pretrain-to-detectron2.py ${WEIGHT_FILE} ${OUTPUT_FILE} # must use .pkl as the output extension.
bash run.sh ${DET_CFG} ${OUTPUT_FILE}
```

## 测试

在训练之后，你可以运行如下命令测试你的模型。

```shell
# distributed version
bash tools/benchmarks/mmdetection/mim_dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS}

# slurm version
bash tools/benchmarks/mmdetection/mim_slurm_test.sh ${PARTITION} ${CONFIG} ${CHECKPOINT}
```

注意：

- `CHECKPOINT`：你想测试的训练好的检测模型。

例子：

```shell
bash ./tools/benchmarks/mmdetection/mim_dist_test.sh \
configs/benchmarks/mmdetection/coco/mask-rcnn_r50_fpn_ms-1x_coco.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 8
```
