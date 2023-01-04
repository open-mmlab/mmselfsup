# Analysis tools

<!-- TOC -->

- [Analysis tools](#analysis-tools)
  - [Count number of parameters](#count-number-of-parameters)
  - [Publish a model](#publish-a-model)
  - [Reproducibility](#reproducibility)
  - [Log Analysis](#log-analysis)

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
