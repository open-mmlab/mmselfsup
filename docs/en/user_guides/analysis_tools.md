# Analysis tools

<!-- TOC -->

- [Analysis tools](#analysis-tools)
  - [Count number of parameters](#count-number-of-parameters)
  - [Publish a model](#publish-a-model)
  - [Reproducibility](#reproducibility)
  - [Log Analysis](#log-analysis)
  - [Visualize Datasets](#visualize-datasets)
  - [Use t-SNE](#use-t-sne)
  - [Pixel Reconstruction Visualization](#pixel-reconstruction-visualization)

## Count number of parameters

```shell
python tools/analysis_tools/count_parameters.py ${CONFIG_FILE}
```

An example:

```shell
python tools/analysis_tools/count_parameters.py configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py
```

## Publish a model

Before you publish a model, you may want to

- Convert model weights to CPU tensors.
- Delete the optimizer states.
- Compute the hash of the checkpoint file and append the hash id to the filename.

```shell
python tools/model_converters/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

An example:

```shell
python tools/model_converters/publish_model.py YOUR/PATH/epoch_100.pth YOUR/PATH/epoch_100_output.pth
```

## Reproducibility

If you want to make your performance exactly reproducible, please set `--cfg-options randomness.deterministic=True` to train the final model. Note that this will switch off `torch.backends.cudnn.benchmark` and slow down the training speed.

## Log Analysis

`tools/analysis_tools/analyze_logs.py` plots loss/lr curves given a training
log file. Run `pip install seaborn` first to install the dependency.

```shell
python tools/analysis_tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

<div align="center">
<img src="https://raw.githubusercontent.com/open-mmlab/mmdetection/master/resources/loss_curve.png" width="400" />
</div>

Examples:

- Plot the classification loss of some run.

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_dense --legend loss_dense
  ```

- Plot the classification and regression loss of some run, and save the figure to a pdf.

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_dense loss_single --out losses.pdf
  ```

- Compare the loss of two runs in the same figure.

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys loss --legend run1 run2
  ```

- Compute the average training speed.

  ```shell
  python tools/analysis_tools/analyze_logs.py cal_train_time log.json [--include-outliers]
  ```

  The output is expected to be like the following.

  ```text
  -----Analyze train time of work_dirs/some_exp/20190611_192040.log.json-----
  slowest epoch 11, average time is 1.2024
  fastest epoch 1, average time is 1.1909
  time std over epochs is 0.0028
  average iter time: 1.1959 s/iter
  ```

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

## Use t-SNE

We provide an off-the-shelf tool to visualize the quality of image representations by t-SNE.

```shell
python tools/analysis_tools/visualize_tsne.py ${CONFIG_FILE} --checkpoint ${CKPT_PATH} --work-dir ${WORK_DIR} [optional arguments]
```

Arguments:

- `CONFIG_FILE`: config file for the pre-trained model.
- `CKPT_PATH`: the path of model's checkpoint.
- `WORK_DIR`: the directory to save the results of visualization.
- `[optional arguments]`: for optional arguments, you can refer to [visualize_tsne.py](https://github.com/open-mmlab/mmselfsup/blob/master/tools/analysis_tools/visualize_tsne.py)

An example:

```shell
python tools/analysis_tools/visualize_tsne.py configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py --checkpoint epoch_100.pth --work-dir work_dirs/selfsup/simsiam_resnet50_8xb32-coslr-200e_in1k
```

An example of visualization:

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/199388251-476a5ad2-f9c1-4dfb-afe2-73cf41b5793b.jpg" width="800" />
</div>

## Pixel Reconstruction Visualization

We provide several pixel reconstruction visualization for listed algorithms:

- MAE
- SimMIM
- MaskFeat

Users can run command below to visualize the reconstruction.

```shell
python tools/analysis_tools/pixel_reconstruction_vis.py ${CONFIG_FILE} --checkpoint ${CKPT_PATH} --img-path ${IMAGE_PATH} --out-file ${OUTPUT_PATH}
```

Arguments:

- `CONFIG_FILE`: config file for the pre-trained model.
- `CKPT_PATH`: the path of model's checkpoint.
- `IMAGE_PATH`: the input image path.
- `OUTPUT_PATH`: the output image path, including 4 sub-images.
- `[optional arguments]`: for optional arguments, you can refer to [pixel_reconstruction_vis.py](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/tools/analysis_tools/pixel_reconstruction_vis.py)

An example:

```shell
python tools/analysis_tools/pixel_reconstruction_vis.py configs/selfsup/mae/mae_vit-huge-p16_8xb512-amp-coslr-1600e_in1k.py --checkpoint https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220916-ff848775.pth --img-path data/imagenet/val/ILSVRC2012_val_00000003.JPEG --out-file test_mae.jpg --norm-pix


# As for SimMIM, it generates the mask in data pipeline, thus we use '--use-vis-pipeline' to apply 'vis_pipeline' defined in config instead of the pipeline defined in script.
python tools/analysis_tools/pixel_reconstruction_vis.py configs/selfsup/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192.py --checkpoint https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192_20220916-4ad216d3.pth --img-path data/imagenet/val/ILSVRC2012_val_00000003.JPEG --out-file test_simmim.jpg --use-vis-pipeline
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
