# Visualization

Visualization can give an intuitive interpretation of the performance of the model.

<!-- TOC -->

- [Visualization](#visualization)
  - [How visualization is implemented](#how-visualization-is-implemented)
  - [What Visualization do in MMSelfsup](#what-visualization-do-in-mmselfsup)
  - [Use different storage backends](#use-different-storage-backends)
  - [Customize Visualization](#customize-visualization)

<!-- /TOC -->

## How visualization is implemented

It is recommended to learn the basic concept of visualization in [engine.md](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/design/visualization.md).

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

The function [`add_datasample()`](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/mmselfsup/visualization/selfsup_visualizer.py#L151) is impleted in [`SelfSupVisualizer`](mmselfsup.visualization.SelfSupVisualizer), and it is mainly used in [browse_dataset.py](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/tools/analysis_tools/browse_dataset.py) for browsing dataset. More tutorial is in [analysis_tools.md](analysis_tools.md)

## Use different storage backends

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

Note that when multiple visualization backends exist for `vis_backends`, only `WandbVisBackend` is valid.

E.g.

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/199388643-288cf83f-0faa-4f34-a5d0-bf53c7bb3e08.png" width="600" />
</div>

## Customize Visualization

The customization of the visualization is similar to other components. If you want to customize `Visualizer`, `VisBackend` or `VisualizationHook`, you can refer to [Visualization Doc](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/visualization.md) in MMEngine.
