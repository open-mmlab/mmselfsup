# Evaluation

<!-- TOC -->

- [Evaluation](#evaluation)
  - [Evaluation in MMEngine](#evaluation-in-mmengine)
    - [Online evaluation](#online-evaluation)
    - [Offline evaluation](#offline-evaluation)
  - [Evaluation In MMSelfSup](#evaluation-in-mmselfsup)
  - [Customize Evaluation](#customize-evaluation)

<!-- /TOC -->

## Evaluation in MMEngine

During model validation and testing, quantitative evaluation is often required. `Metric` and `Evaluator` have been implemented in MMEngine to perform this function. See [MMEngine Doc](https://mmengine.readthedocs.io/en/latest/design/evaluation.html).

Model evaluation is divided into online evaluation and offline evaluation.

### Online evaluation

Online evaluation is used in [`ValLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L300) and [`TestLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L373).

Take `ValLoop` for example:

```python
...
class ValLoop(BaseLoop):
    ...
    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        ...
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
```

### Offline evaluation

Offline evaluation uses the predictions saved in a file. In this case, since there is no `Runner`, we need to build the `Evaluator` and call `offline_evaluate()` function.

An example:

```python
from mmengine.evaluator import Evaluator
from mmengine.fileio import load

evaluator = Evaluator(metrics=dict(type='Accuracy', top_k=(1, 5)))

data = load('test_data.pkl')
predictions = load('prediction.pkl')

results = evaluator.offline_evaluate(data, predictions, chunk_size=128)
```

## Evaluation In MMSelfSup

During pretrain, validation and testing are not included, so it is no need to use evaluation.

During benchmark, the pre-trained models need other downstream tasks to evaluate the performance, e.g. classification, detection, segmentation, etc. It is recommended to run downstream tasks with other OpenMMLab repos, such as MMClassification or MMDetection, which have already implemented their own evaluation functionalities.

But MMSelfSup also implements some custom evaluation functionalities to support downstream tasks, shown as below:

- [`knn_classifier()`](mmselfsup.evaluation.functional.knn_classifier)

It compute accuracy of knn classifier predictions, and is used in [KNN evaluation](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/tools/benchmarks/classification/knn_imagenet/test_knn.py#L179).

```python
...
top1, top5 = knn_classifier(train_feats, train_labels, val_feats,
                                        val_labels, k, args.temperature)
...
```

- [`ResLayerExtraNorm`](mmselfsup.evaluation.functional.ResLayerExtraNorm)

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

Custom `Metric` and `Evaluator`  are also supported, see [MMEngine Doc](https://mmengine.readthedocs.io/en/latest/design/evaluation.html)
