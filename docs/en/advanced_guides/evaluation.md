# Evaluation

<!-- TOC -->

- [Evaluation](#evaluation)
  - [Evaluation in MMEngine](#evaluation-in-mmengine)
  - [Evaluation In MMSelfSup](#evaluation-in-mmselfsup)
  - [Customize Evaluation](#customize-evaluation)

<!-- /TOC -->

## Evaluation in MMEngine

During model validation and testing, quantitative evaluation is often required. `Metric` and `Evaluator` have been implemented in MMEngine to perform this function. See [MMEngine Doc](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/design/metric_and_evaluator.md).

Model evaluation is divided into online evaluation and offline evaluation, as shown in the following figure:

![img](https://user-images.githubusercontent.com/15977946/163718224-20a4970a-e540-4a3a-8b01-bf0a604c6841.jpg)

## Evaluation In MMSelfSup

During pretrain, validation and testing are not included, so it is no need to use evaluation.

During benchmark, evaluation is used on downstream tasks with other OpenMMLab projects like `MMDetection` and `MMSegmentation` .

Some evaluation functions are impleted in MMSelfSup, as follows:

- [`knn_classifier()`](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/mmselfsup/evaluation/functional/knn_classifier.py)

It compute accuracy of knn classifier predictions, and is used in [KNN evaluation](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/tools/benchmarks/classification/knn_imagenet/test_knn.py#L179).

```python
...
top1, top5 = knn_classifier(train_feats, train_labels, val_feats,
                                        val_labels, k, args.temperature)
...
```

- [`ResLayerExtraNorm`](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/mmselfsup/evaluation/functional/res_layer_extra_norm.py)

It add extra norm to original `ResLayer`, and is used in mmdetection benchmark config.

```python
model = dict(
    backbone=...,
    roi_head=dict(
        shared_head=dict(
            type='ResLayerExtraNorm',
            norm_cfg=norm_cfg,
            norm_eval=False,
            style='pytorch')))
```

## Customize Evaluation

Custom `Metric` and `Evaluator`  are also supported, see [MMEngine Doc](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/evaluation.md)
