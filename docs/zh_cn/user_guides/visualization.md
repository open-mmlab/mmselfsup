# 可视化

可视化能直观反映模型性能表现。

<!-- TOC -->

- [可视化](#可视化)
  - [如何实现可视化](#如何实现可视化)
  - [MMSelfSup 的可视化做什么](#mmselfsup-的可视化做什么)
  - [用不同的存储后端](#用不同的存储后端)
  - [定制化的可视化](#定制化的可视化)
  - [数据集可视化](#数据集可视化)
  - [t-SNE 可视化](#t-sne-可视化)
  - [可视化低级特征重建](#可视化低级特征重建)
  - [可视化 shape bias](#可视化-shape-bias)
    - [准备数据集](#准备数据集)
    - [为分类调整配置](#为分类调整配置)
    - [用上述调整过的配置文件推理模型](#用上述调整过的配置文件推理模型)
    - [画出 shape bias](#画出-shape-bias)

<!-- /TOC -->

## 如何实现可视化

建议先学习 [文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/design/visualization.md) 里关于可视化的基本概念。

OpenMMLab 2.0 引入可视化对象 `Visualizer` 和一些可视化后端 `VisBackend` 。如下图表展示了 `Visualizer` 和 `VisBackend` 的关系。

<div align="center">
<img src="https://user-images.githubusercontent.com/17425982/163327736-f7cb3b16-ef07-46bc-982a-3cc7495e6c82.png" width="800" />
</div>

## MMSelfSup 的可视化做什么

(1) 用不同的存储后端存训练数据

MMEngine 的后端包括 `LocalVisBackend`, `TensorboardVisBackend` 和 `WandbVisBackend`。

在训练过程中，默认钩子 `LoggerHook` 中的 [after_train_iter()](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py#L150) 会被调用，并且会在不同后端中用到 `add_scalars`，例如：

```python
...
def after_train_iter(...):
    ...
    runner.visualizer.add_scalars(
        tag, step=runner.iter + 1, file_path=self.json_log_path)
...
```

(2) 浏览数据集

[`add_datasample()`](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/mmselfsup/visualization/selfsup_visualizer.py#L151) 函数位于 [`SelfSupVisualizer`](mmselfsup.visualization.SelfSupVisualizer), 常用于在 [browse_dataset.py](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/tools/analysis_tools/browse_dataset.py) 中浏览数据集。更多细节可以参考 [数据集可视化](#数据集可视化)。

## 用不同的存储后端

如果想用不同的存储后端( Wandb, Tensorboard, 或者远程窗口里常规的后端)，像以下这样改配置文件的 `vis_backends` 就行了：

**Local**

```python
vis_backends = [dict(type='LocalVisBackend')]
```

**Tensorboard**

```python
vis_backends = [dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SelfSupVisualizer', vis_backends=vis_backends, name='visualizer')
```

例如

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/199388357-5d1cc7b4-07b8-41b1-ac66-12ec8ef009da.png" width="400" />
</div>

**Wandb**

```python
vis_backends = [dict(type='WandbVisBackend')]
visualizer = dict(
    type='SelfSupVisualizer', vis_backends=vis_backends, name='visualizer')
```

例如：

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/199388643-288cf83f-0faa-4f34-a5d0-bf53c7bb3e08.png" width="600" />
</div>

## 定制化的可视化

定制化可视化就像定制化其他组成部分那样。想定制化 `Visualizer`, `VisBackend` 或者 `VisualizationHook` 的话可以参考 MMEngine 里的 [可视化文档](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/visualization.md)

## 数据集可视化

`tools/misc/browse_dataset.py` 帮助用户可视化浏览 MMSelfSup 数据集，或者也可以把图像存到指定的目录里。

```shell
python tools/misc/browse_dataset.py ${CONFIG} [-h] [--skip-type ${SKIP_TYPE[SKIP_TYPE...]}] [--output-dir ${OUTPUT_DIR}] [--not-show] [--show-interval ${SHOW_INTERVAL}]
```

例子如下：

```shell
python tools/misc/browse_dataset.py configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py
```

一个可视化的例子如下：

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/199387454-219e6f6c-fbb7-43bb-b319-61d3e6266abc.png" width="600" />
</div>

- 左边两张图来自对比学习数据流。
- 右边那张图是添加了掩码的图像。

## t-SNE 可视化

我们提供可视化 t-SNE 展示图片表征的现成工具。

```shell
python tools/analysis_tools/visualize_tsne.py ${CONFIG_FILE} --checkpoint ${CKPT_PATH} --work-dir ${WORK_DIR} [optional arguments]
```

参数:

- `CONFIG_FILE`: 位于 `configs/tsne/` 中的 t-SNE 的配置文件。
- `CKPT_PATH`: 模型检查点的目录或链接。
- `WORK_DIR`: 拿来存可视化结果的目录。
- `[optional arguments]`: 可选项，可以参考 [visualize_tsne.py](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/tools/analysis_tools/visualize_tsne.py)

一个命令示例如下：

```shell
python ./tools/analysis_tools/visualize_tsne.py \
    configs/tsne/resnet50_imagenet.py \
    --checkpoint https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/mocov2_resnet50_8xb32-coslr-200e_in1k_20220825-b6d23c86.pth \
    --work-dir  ./work_dirs/tsne/mocov2/ \
    --max-num-class 100
```

下面是可视化的例子,左边来自 `MoCoV2_ResNet50`，右边来自 `MAE_ViT-base`:

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/207305086-91df298c-0eb7-4254-9c5b-ba711644501b.png" width="250" />
<img src="https://user-images.githubusercontent.com/36138628/207305333-59af4747-1e9c-4f85-a57d-c7e5d132a6e5.png" width="250" />
</div>

## 可视化低级特征重建

我们提供如下算法的重建可视化：

- MAE
- SimMIM
- MaskFeat

用户可以通过如下命令可视化重建。

```shell
python tools/analysis_tools/visualize_reconstruction.py ${CONFIG_FILE} \
    --checkpoint ${CKPT_PATH} \
    --img-path ${IMAGE_PATH} \
    --out-file ${OUTPUT_PATH}
```

参数:

- `CONFIG_FILE`: 预训练模型配置文件。
- `CKPT_PATH`: 模型检查点的路径。
- `IMAGE_PATH`: 输入图像的路径。
- `OUTPUT_PATH`: 输出图像的路径，包含４个子图。
- `[optional arguments]`: for optional arguments, 您可以参考 [visualize_reconstruction.py](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/tools/analysis_tools/visualize_reconstruction.py) 了解可选参数。

例子如下:

```shell
python tools/analysis_tools/visualize_reconstruction.py configs/selfsup/mae/mae_vit-huge-p16_8xb512-amp-coslr-1600e_in1k.py \
    --checkpoint https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220916-ff848775.pth \
    --img-path data/imagenet/val/ILSVRC2012_val_00000003.JPEG \
    --out-file test_mae.jpg \
    --norm-pix


# SimMIM 在数据流里生成掩码，所以我们不用脚本里定义好的管道而用 '--use-vis-pipeline' 来应用配置里定义的 'vis_pipeline'
python tools/analysis_tools/visualize_reconstruction.py configs/selfsup/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192.py \
    --checkpoint https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192_20220916-4ad216d3.pth \
    --img-path data/imagenet/val/ILSVRC2012_val_00000003.JPEG \
    --out-file test_simmim.jpg \
    --use-vis-pipeline
```

MAE 结果如下:

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/200465826-83f316ed-5a46-46a9-b665-784b5332d348.jpg" width="800" />
</div>

SimMIM 结果如下:

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/200466133-b77bc9af-224b-4810-863c-eed81ddd1afa.jpg" width="800" />
</div>

MaskFeat 结果如下:

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/200465876-7e7dcb6f-5e8d-4d80-b300-9e1847cb975f.jpg" width="800" />
</div>

## 可视化 shape bias

shape bias 衡量在感知图像特征的过程中，与纹理相比，模型依赖 shape 的程度。感兴趣的话可以参考 [paper](https://arxiv.org/abs/2106.07411) 了解更多信息。 MMSelfSup 提供一个现有的用于得到分类模型 shape bias 的工具箱。可以按以下步骤来做：

### 准备数据集

首先把 [cue-conflict](https://github.com/bethgelab/model-vs-human/releases/download/v0.1/cue-conflict.tar.gz) 下载到 `data` 文件夹里,然后解压数据集。然后，您的 `data` 文件夹的结构应该像这样：

```text
data
├──cue-conflict
|      |──airplane
|      |──bear
|      ...
|      |── truck
```

### 为分类调整配置

用以下配置代替原来的 test_dataloader 和 test_evaluation

```python
test_dataloader = dict(
    dataset=dict(
        type='CustomDataset',
        data_root='data/cue-conflict',
        _delete_=True),
    drop_last=False)
test_evaluator = dict(
    type='mmselfsup.ShapeBiasMetric',
    _delete_=True,
    csv_dir='directory/to/save/the/csv/file',
    model_name='your_model_name')
```

请记得自己修改一下 `csv_dir` 和 `model_name`。

### 用上述调整过的配置文件推理模型

然后您需要做的是用调整过的配置文件在 `cue-conflict` 数据集上推理模型。

```shell
# For Slurm
GPUS_PER_NODE=1 GPUS=1 bash tools/benchmarks/classification/mim_slurm_test.sh $partition $config $checkpoint
```

```shell
# For PyTorch
GPUS=1 bash tools/benchmarks/classification/mim_dist_test.sh $config $checkpoint
```

在这之后，可以获得名为 `cue-conflict_model-name_session-1.csv` 的 csv 文件。除了这个文件之外，您应该下载 [csv 文件](https://github.com/bethgelab/model-vs-human/tree/master/raw-data/cue-conflict) 到对应的 `csv_dir`。

### 画出 shape bias

然后我们就可以开始画出 shape bias 了。

```shell
python tools/analysis_tools/visualize_shape_bias.py --csv-dir $CVS_DIR --result-dir $CSV_DIR --colors $RGB --markers o --plotting-names $YOU_MODEL_NAME --model-names $YOU_MODEL_NAME
```

- `--csv-dir`, 相同目录下，用于存储 csv 文件。
- `--colors`, 应为以 RGB 为格式的 RGB 值,比如 100 100 100,如果您想画若干模型的 shape bias 的话多个RGB值也行。
- `--plotting-names`, 偏好形状里图例的名称，您可将之设为模型名字。如果您想画若干模型的 shape bias 的话名字设多个值也行。
- `--model-names`，应该跟配置文件里的一样，如果您想画若干模型的 shape bias 的话多个名字也行。

请注意，每三个 `--colors` 对应一个 `--model-names` 。上面步骤做完后您会得到如下图像：

<div align="center">
<img src="https://user-images.githubusercontent.com/30762564/208357938-c744d3c3-7e08-468e-82b7-fc5f1804da59.png" width="400" />
</div>
