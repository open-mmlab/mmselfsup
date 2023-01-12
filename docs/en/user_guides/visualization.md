# Visualization

Visualization can give an intuitive interpretation of the performance of the model.

<!-- TOC -->

- [Visualization](#visualization)
  - [How visualization is implemented](#how-visualization-is-implemented)
  - [What Visualization do in MMSelfsup](#what-visualization-do-in-mmselfsup)
  - [Use Different Storage Backends](#use-different-storage-backends)
  - [Customize Visualization](#customize-visualization)
  - [Visualize Datasets](#visualize-datasets)
  - [Visualize t-SNE](#visualize-t-sne)
  - [Visualize Low-level Feature Reconstruction](#visualize-low-level-feature-reconstruction)
  - [Visualize Shape Bias](#visualize-shape-bias)
    - [Prepare the dataset](#prepare-the-dataset)
    - [Modify the config for classification](#modify-the-config-for-classification)
    - [Inference your model with above modified config file](#inference-your-model-with-above-modified-config-file)
    - [Plot shape bias](#plot-shape-bias)

<!-- /TOC -->

## How visualization is implemented

It is recommended to learn the basic concept of visualization in [documentation](https://github.com/open-mmlab/mmengine/blob/main/docs/en/design/visualization.md).

OpenMMLab 2.0 introduces the visualization object `Visualizer` and several visualization backends `VisBackend`. The diagram below shows the relationship between `Visualizer` and  `VisBackend`,

<div align="center">
<img src="https://user-images.githubusercontent.com/17425982/163327736-f7cb3b16-ef07-46bc-982a-3cc7495e6c82.png" width="800" />
</div>

## What Visualization do in MMSelfsup

(1) Save training data using different storage backends

The backends in MMEngine includes `LocalVisBackend`, `TensorboardVisBackend` and `WandbVisBackend` .

During training,  [after_train_iter()](https://github.com/open-mmlab/mmengine/blob/main/mmengine/hooks/logger_hook.py#L150) in the default hook `LoggerHook` will be called, and use `add_scalars` in different backends, as follows:

```python
...
def after_train_iter(...):
    ...
    runner.visualizer.add_scalars(
        tag, step=runner.iter + 1, file_path=self.json_log_path)
...
```

(2) Browse dataset

The function [`add_datasample()`](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/mmselfsup/visualization/selfsup_visualizer.py#L151) is impleted in [`SelfSupVisualizer`](mmselfsup.visualization.SelfSupVisualizer), and it is mainly used in [browse_dataset.py](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/tools/analysis_tools/browse_dataset.py) for browsing dataset. More tutorial is in section [Visualize Datasets](#visualize-datasets)

## Use Different Storage Backends

If you want to use a different backend (Wandb, Tensorboard, or a custom backend with a remote window), just change the `vis_backends` in the config, as follows:

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

E.g.

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/199388357-5d1cc7b4-07b8-41b1-ac66-12ec8ef009da.png" width="400" />
</div>

**Wandb**

```python
vis_backends = [dict(type='WandbVisBackend')]
visualizer = dict(
    type='SelfSupVisualizer', vis_backends=vis_backends, name='visualizer')
```

E.g.

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/199388643-288cf83f-0faa-4f34-a5d0-bf53c7bb3e08.png" width="600" />
</div>

## Customize Visualization

The customization of the visualization is similar to other components. If you want to customize `Visualizer`, `VisBackend` or `VisualizationHook`, you can refer to [Visualization Doc](https://github.com/open-mmlab/mmengine/blob/main/docs/en/advanced_tutorials/visualization.md) in MMEngine.

## Visualize Datasets

`tools/misc/browse_dataset.py` helps the user to browse a mmselfsup dataset (transformed images) visually, or save the image to a designated directory.

```shell
python tools/misc/browse_dataset.py ${CONFIG} [-h] [--skip-type ${SKIP_TYPE[SKIP_TYPE...]}] [--output-dir ${OUTPUT_DIR}] [--not-show] [--show-interval ${SHOW_INTERVAL}]
```

An example:

```shell
python tools/misc/browse_dataset.py configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py
```

An example of visualization:

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/199387454-219e6f6c-fbb7-43bb-b319-61d3e6266abc.png" width="600" />
</div>

- The left two pictures are images from contrastive learning data pipeline.
- The right one is a masked image.

## Visualize t-SNE

We provide an off-the-shelf tool to visualize the quality of image representations by t-SNE.

```shell
python tools/analysis_tools/visualize_tsne.py ${CONFIG_FILE} --checkpoint ${CKPT_PATH} --work-dir ${WORK_DIR} [optional arguments]
```

Arguments:

- `CONFIG_FILE`: config file for t-SNE, which listed in the directory `configs/tsne/`
- `CKPT_PATH`: the path or link of the model's checkpoint.
- `WORK_DIR`: the directory to save the results of visualization.
- `[optional arguments]`: for optional arguments, you can refer to [visualize_tsne.py](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/tools/analysis_tools/visualize_tsne.py)

An example of command:

```shell
python ./tools/analysis_tools/visualize_tsne.py \
    configs/tsne/resnet50_imagenet.py \
    --checkpoint https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/mocov2_resnet50_8xb32-coslr-200e_in1k_20220825-b6d23c86.pth \
    --work-dir  ./work_dirs/tsne/mocov2/ \
    --max-num-class 100
```

An example of visualization, left is from `MoCoV2_ResNet50` and right is from `MAE_ViT-base`:

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/207305086-91df298c-0eb7-4254-9c5b-ba711644501b.png" width="250" />
<img src="https://user-images.githubusercontent.com/36138628/207305333-59af4747-1e9c-4f85-a57d-c7e5d132a6e5.png" width="250" />
</div>

## Visualize Low-level Feature Reconstruction

We provide several reconstruction visualization for listed algorithms:

- MAE
- SimMIM
- MaskFeat

Users can run command below to visualize the reconstruction.

```shell
python tools/analysis_tools/visualize_reconstruction.py ${CONFIG_FILE} \
    --checkpoint ${CKPT_PATH} \
    --img-path ${IMAGE_PATH} \
    --out-file ${OUTPUT_PATH}
```

Arguments:

- `CONFIG_FILE`: config file for the pre-trained model.
- `CKPT_PATH`: the path of model's checkpoint.
- `IMAGE_PATH`: the input image path.
- `OUTPUT_PATH`: the output image path, including 4 sub-images.
- `[optional arguments]`: for optional arguments, you can refer to [visualize_reconstruction.py](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/tools/analysis_tools/visualize_reconstruction.py)

An example:

```shell
python tools/analysis_tools/visualize_reconstruction.py configs/selfsup/mae/mae_vit-huge-p16_8xb512-amp-coslr-1600e_in1k.py \
    --checkpoint https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220916-ff848775.pth \
    --img-path data/imagenet/val/ILSVRC2012_val_00000003.JPEG \
    --out-file test_mae.jpg \
    --norm-pix


# As for SimMIM, it generates the mask in data pipeline, thus we use '--use-vis-pipeline' to apply 'vis_pipeline' defined in config instead of the pipeline defined in script.
python tools/analysis_tools/visualize_reconstruction.py configs/selfsup/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192.py \
    --checkpoint https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192_20220916-4ad216d3.pth \
    --img-path data/imagenet/val/ILSVRC2012_val_00000003.JPEG \
    --out-file test_simmim.jpg \
    --use-vis-pipeline
```

Results of MAE:

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/200465826-83f316ed-5a46-46a9-b665-784b5332d348.jpg" width="800" />
</div>

Results of SimMIM:

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/200466133-b77bc9af-224b-4810-863c-eed81ddd1afa.jpg" width="800" />
</div>

Results of MaskFeat:

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/200465876-7e7dcb6f-5e8d-4d80-b300-9e1847cb975f.jpg" width="800" />
</div>

## Visualize Shape Bias

Shape bias measures how a model relies the shapes, compared to texture, to sense the semantics in images. For more details,
we recommend interested readers to this [paper](https://arxiv.org/abs/2106.07411). MMSelfSup provide an off-the-shelf toolbox to
obtain the shape bias of a classification model. You can following these steps below:

### Prepare the dataset

First you should download the [cue-conflict](https://github.com/bethgelab/model-vs-human/releases/download/v0.1/cue-conflict.tar.gz) to `data` folder,
and then unzip this dataset. After that, you `data` folder should have the following structure:

```text
data
├──cue-conflict
|      |──airplane
|      |──bear
|      ...
|      |── truck
```

### Modify the config for classification

Replace the original test_dataloader and test_evaluation with following configurations

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

Please note you should make custom modifications to the `csv_dir` and `model_name`.

### Inference your model with above modified config file

Then you should inferece your model on the `cue-conflict` dataset with the your modified config files.

```shell
# For Slurm
GPUS_PER_NODE=1 GPUS=1 bash tools/benchmarks/classification/mim_slurm_test.sh $partition $config $checkpoint
```

```shell
# For PyTorch
GPUS=1 bash tools/benchmarks/classification/mim_dist_test.sh $config $checkpoint
```

After that, you should obtain a csv file, named `cue-conflict_model-name_session-1.csv`. Besides this file, you should
also download these [csv files](https://github.com/bethgelab/model-vs-human/tree/master/raw-data/cue-conflict) to the
`csv_dir`.

### Plot shape bias

Then we can start to plot the shape bias

```shell
python tools/analysis_tools/visualize_shape_bias.py --csv-dir $CVS_DIR --result-dir $CSV_DIR --colors $RGB --markers o --plotting-names $YOU_MODEL_NAME --model-names $YOU_MODEL_NAME
```

- `--csv-dir`, the same directory to save these csv files
- `--colors`, should be the RGB values, formatted in R G B, e.g. 100 100 100, and can be multiple RGB values, if you want
  to plot the shape bias of several models
- `--plotting-names`, the name of the legend in the shape bias figure, and you can set it as your model name. If you want
  to plot several models, plotting_names can be multiple values
- `--model-names`, should be the same name specified in your config, and can be multiple names if you want to plot the shape bias of several models

Please note, every three values for `--colors` corresponds to one value for `--model-names`. After all of above steps, you
are expected to obtain the following figure.

<div align="center">
<img src="https://user-images.githubusercontent.com/30762564/208357938-c744d3c3-7e08-468e-82b7-fc5f1804da59.png" width="400" />
</div>
