# 模型评测

<!-- TOC -->

- [模型评测](#模型评测)
  - [MMEngine 中的评测](#mmengine-中的评测)
    - [在线评测](#在线评测)
    - [离线评测](#离线评测)
  - [MMSelfSup 中的评测](#mmselfsup-中的评测)
  - [自定义评测](#自定义评测)

<!-- /TOC -->

## MMEngine 中的评测

在模型验证和测试过程中，经常需要进行定量评估. 在MMEngine中已经实现了 `Metric` 和 `Evaluator`  来执行此功能. 详情请见[MMEngine Doc](https://mmengine.readthedocs.io/en/latest/design/evaluation.html).

模型评估分为在线评测和离线评测.

### 在线评测

在线评测用于 [`ValLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L300) 和 [`TestLoop`](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L373) 中.

以 `ValLoop` 为例:

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

### 离线评测

离线评测使用保存在文件中的预测结果.在这种情况下, 由于没有 `Runner`, 我们需要构建 `Evaluator` 并调用 `offline_evaluate()` 函数.

一个例子:

```python
from mmengine.evaluator import Evaluator
from mmengine.fileio import load

evaluator = Evaluator(metrics=dict(type='Accuracy', top_k=(1, 5)))

data = load('test_data.pkl')
predictions = load('prediction.pkl')

results = evaluator.offline_evaluate(data, predictions, chunk_size=128)
```

## MMSelfSup 中的评测

在预训练期间，因为不包含验证和测试，所以不需要使用模型评测.

在基准测试期间, 预训练模型需要 `classification`、 `detection`、 `segmentation` 等其他的下游任务来评测其性能. 建议用其他的 OpenMMLab 仓库运行下游任务, 例如 `MMClassification` 或 `MMDetection`, 它们已经实现了自己评估功能.

但是 MMSelfSup 也实现了某些自定义的评测功能去支持下游任务, 如下所示:

- [`knn_classifier()`](mmselfsup.evaluation.functional.knn_classifier)

用于计算 knn 分类器预测的准确性,并且用于 [KNN evaluation](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/tools/benchmarks/classification/knn_imagenet/test_knn.py#L179).

```python
...
top1, top5 = knn_classifier(train_feats, train_labels, val_feats,
                                        val_labels, k, args.temperature)
...
```

- [`ResLayerExtraNorm`](mmselfsup.evaluation.functional.ResLayerExtraNorm)

为原始的 `ResLayer` 增加了额外的规范, 并在mmdetection基准配置中使用.

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

## 自定义评测

支持 `Metric` 和 `Evaluator` 的自定义评测,详情请见 [MMEngine Doc](https://mmengine.readthedocs.io/en/latest/design/evaluation.html)
