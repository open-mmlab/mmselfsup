# 分析工具

<!-- TOC -->

- [分析工具](#分析工具)
  - [统计参数量](#统计参数量)
  - [发布模型](#发布模型)
  - [结果复现](#结果复现)
  - [日志分析](#日志分析)

## 统计参数量

```shell
python tools/analysis_tools/count_parameters.py ${CONFIG_FILE}
```

一个例子如下：

```shell
python tools/analysis_tools/count_parameters.py configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py
```

## 发布模型

发布模型之前，你可能是想：

- 把模型权重转换为 CPU 张量。
- 删除优化器相关状态。
- 计算检查点文件的哈希值并把哈希 ID 加到文件名上。

```shell
python tools/model_converters/publish_model.py ${INPUT_FILENAME} ${OUTPUT_FILENAME}
```

例子如下：

```shell
python tools/model_converters/publish_model.py YOUR/PATH/epoch_100.pth YOUR/PATH/epoch_100_output.pth
```

## 结果复现

想让你的结果完全可以复现的话，训练最终模型时请设置 `--cfg-options randomness.deterministic=True` 。值得一提的是，这会关掉 `torch.backends.cudnn.benchmark` 并降低训练速度。

## 日志分析

`tools/analysis_tools/analyze_logs.py` 用训练日志文件画损失/学习率曲线。首先 `pip install seaborn` 安装依赖库。

```shell
python tools/analysis_tools/analyze_logs.py plot_curve [--keys ${KEYS}] [--title ${TITLE}] [--legend ${LEGEND}] [--backend ${BACKEND}] [--style ${STYLE}] [--out ${OUT_FILE}]
```

<div align="center">
<img src="https://raw.githubusercontent.com/open-mmlab/mmdetection/master/resources/loss_curve.png" width="400" />
</div>

例子如下:

- 画部分运行过程中分类的损失函数图像。

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_dense --legend loss_dense
  ```

- 画部分运行过程中分类和倒退的损失函数图像并存到 pdf 文件里。

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log.json --keys loss_dense loss_single --out losses.pdf
  ```

- 在同一张图内，比较两次训练的损失。

  ```shell
  python tools/analysis_tools/analyze_logs.py plot_curve log1.json log2.json --keys loss --legend run1 run2
  ```

- 计算平均训练速度。

  ```shell
  python tools/analysis_tools/analyze_logs.py cal_train_time log.json [--include-outliers]
  ```

  输出应该像下面这样：

  ```text
  -----Analyze train time of work_dirs/some_exp/20190611_192040.log.json-----
  slowest epoch 11, average time is 1.2024
  fastest epoch 1, average time is 1.1909
  time std over epochs is 0.0028
  average iter time: 1.1959 s/iter
  ```
