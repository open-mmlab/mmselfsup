# Visualization

可视化可以给深度学习的模型训练提供直观解释。

<!-- TOC -->

- [Visualization](#visualization)
  - [可视化是如何实现的](#可视化是如何实现的)
  - [可视化有哪些功能](#可视化有哪些功能)
  - [怎么使用不同的存储后端](#怎么使用不同的存储后端)
  - [如何自定义可视化](#如何自定义可视化)

<!-- /TOC -->


## 可视化是如何实现的

OpenMMLab 2.0 引入了可视化对象 Visualizer 和各个可视化存储后端 VisBackend 如 `LocalVisBackend`、`WandbVisBackend` 和 `TensorboardVisBackend` 等。此处的可视化不仅仅包括图片数据格式，还包括配置内容、标量和模型图等数据的可视化。

- 为了方便调用，Visualizer 提供的接口实现了绘制和存储的功能。可视化存储后端 VisBackend 作为 Visualizer 的内部属性，会在需要的时候被 Visualizer 调用，将数据存到不同的后端
- 考虑到绘制后会希望存储到多个后端，Visualizer 可以配置多个 VisBackend，当用户调用 Visualizer 的存储接口时候，Visualizer 内部会遍历的调用 VisBackend 存储接口


SelfSupVisualizer 继承了 mmengine 的 [Visualizer]()，并添加了add_datasample()函数，这主要用于 tools/misc/browse_dataset.py.

见engine的文档

## 可视化有哪些功能


（１）使用不同的存储后端来保存训练数据，除了在 LoggerHook 中默认调用 [add_scalars()]() 方法,

调用默认钩子 LoggerHook 的 函数 after_train_iter()，如下所示：
```python
runner.visualizer.add_scalars(
    tag, step=runner.iter + 1, file_path=self.json_log_path)
```
具体用法在下面

（２）我们还支持 browse_dataset.py, 展示通过transforms之后的dataset。具体用法及介绍可以参考 user_guides/analysis_tools.md

SelfSupVisualizer 添加了add_datasample()函数，这主要用于 tools/misc/browse_dataset.py.


## 怎么使用不同的存储后端

用户可以指定 Wandb 、Tensorboard 或者自定义具备远程窗口显示的后端来保存数据，然后在浏览器上显示。
如果想使用不同的存储后端，只需要修改 config 中的 vis_backends, 如下所示：

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

**Wandb**

使用方法和上面完全一致。需要特别注意的是由于 Wandb 绘制的数据无法和 `LocalVisBackend` 后端兼容，所以当 `vis_backends` 存在多个可视化存储后端时候只有 `WandbVisBackend` 才是有效的。

```python
vis_backends = [dict(type='WandbVisBackend')]
visualizer = dict(
    type='SelfSupVisualizer', vis_backends=vis_backends, name='visualizer')
```



## 如何自定义可视化

自定义可视化的方法跟其他组件相似，但是目前在ＭＭＥngine的功能已经可以覆盖MMSelfsup的用法，因此没有做过多的设计。









