# Engine

<!-- TOC -->

- [Engine](#engine)
  - [Hook](#hook)
    - [Introduction](#introduction)
    - [Implemented hooks in MMSelfsup](#implemented-hooks-in-mmselfsup)
    - [An example](#an-example)
  - [Optimizer](#optimizer)
    - [Introduction](#introduction-1)
    - [Implemented optimizers in MMSelfsup](#implemented-optimizers-in-mmselfsup)
  - [Parameter Scheduler](#parameter-scheduler)
    - [Introduction](#introduction-2)
    - [Used Schedulers in MMSelfsup](#used-schedulers-in-mmselfsup)
    - [An example](#an-example-1)

<!-- /TOC -->



## Hook

### Introduction



**(1) Why Use Hook**

Hook programming is a programming model in which a bitpoint (mount point) is set in one or more locations of a program, and when the program runs to a bitpoint, all methods registered to the bitpoint at runtime are automatically called. Hook programming can improve the flexibility and extensibility of the program, and the user can register the custom methods to the loci and they can be called without modifying the code in the program.



**(2) What is the difference between Hook in mmengine and pytorch**

Hooks are also used everywhere in PyTorch, for example in the neural network module (nn.Module) to get the forward input and output of the module as well as the reverse input and output. Take the [`register_forward_hook`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook) method For example, this method registers a forward hook with the module, and the hook can get the forward input and output of the module.

In MMEngine, we abstract the training process into an executor (Runner). Besides initializing the environment, another function of the executor is to call the hook at specific loci to complete the customization logic. For more information about executors, please read [Executor Documentation](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html).

For easy management, MMEngine defines loci as methods and integrates them into [Hook base class (Hook)](https://mmengine.readthedocs.io/zh/latest/api.html#hook), we just need to inherit the Hook base class and implement custom logic at specific loci according to our needs, and then register the hooks then register the hook to the executor, and the methods of the corresponding loci in the hook can be called automatically.



**(3) Usage of Hooks in mmengine**

Hooks only work after being registered into the runner. At present, hooks are mainly divided into two categories:

- Built-in hooks

  - Default hooks

  - Custom hooks

Those hooks are registered by the runner by default. Generally, they fulfill some basic functions, and have default priority, you don't need to modify the priority.

- custom hooks

The custom hooks are registered through custom_hooks. Generally, they are hooks with enhanced functions. The priority needs to be specified in the configuration file. If you do not specify the priority of the hook, it will be set to 'NORMAL' by default.

The hook mechanism is widely used in the OpenMMLab open-source algorithm library. Inserted in the To learn more, please refer to [MMEngine Hook Docs](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/hook.md)



### Implemented hooks in MMSelfsup

Some hooks have been already implemented in MMSelfsup, they are:

- [DeepClusterHook](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/mmselfsup/engine/hooks/deepcluster_hook.py)

- [DenseCLHook](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/mmselfsup/engine/hooks/densecl_hook.py)

- [ODCHook](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/mmselfsup/engine/hooks/odc_hook.py)

- [SimSiamHook](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/mmselfsup/engine/hooks/simsiam_hook.py)

- [SwAVHook](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/mmselfsup/engine/hooks/swav_hook.py)

- ......




### An example

`loss_lambda` is Loss weight for the single and dense contrastive loss. Defaults to 0.5.


```python
losses = dict()
losses['loss_single'] = loss_single * (1 - self.loss_lambda)
losses['loss_dense'] = loss_dense * self.loss_lambda
```

The `loss_lambda` warmup is in `mmselfsup/engine/hooks/densecl_hook.py`:


```python
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

from mmengine.hooks import Hook

from mmselfsup.registry import HOOKS
from mmselfsup.utils import get_model


@HOOKS.register_module()
class DenseCLHook(Hook):
    """Hook for DenseCL.

    This hook includes ``loss_lambda`` warmup in DenseCL.
    Borrowed from the authors' code: `<https://github.com/WXinlong/DenseCL>`_.

    Args:
        start_iters (int): The number of warmup iterations to set
            ``loss_lambda=0``. Defaults to 1000.
    """

    def __init__(self, start_iters: int = 1000) -> None:
        self.start_iters = start_iters

    def before_train(self, runner) -> None:
        """Obtain ``loss_lambda`` from algorithm."""
        assert hasattr(get_model(runner.model), 'loss_lambda'), \
            "The runner must have attribute \"loss_lambda\" in DenseCL."
        self.loss_lambda = get_model(runner.model).loss_lambda

    def before_train_iter(self,
                          runner,
                          batch_idx: int,
                          data_batch: Optional[Sequence[dict]] = None) -> None:
        """Adjust ``loss_lambda`` every train iter."""
        assert hasattr(get_model(runner.model), 'loss_lambda'), \
            "The runner must have attribute \"loss_lambda\" in DenseCL."
        cur_iter = runner.iter
        if cur_iter >= self.start_iters:
            get_model(runner.model).loss_lambda = self.loss_lambda
        else:
            get_model(runner.model).loss_lambda = 0.

```




## Optimizer

### Introduction



**(1) Why use an optimizer**

During model training, we need to use optimization algorithms to optimize the parameters of the model.

PyTorch's `torch.optim` contains implementations of various optimization algorithms, and the classes of these optimization algorithms are called optimizers. A detailed description of PyTorch optimizers can be found in the [PyTorch Optimizer Documentation](https://pytorch.org/docs/stable/optim.html#)



**(2) What is the difference between the optimizer in mmengine and that in pytorch**

- MMEngine supports all PyTorch optimizers, and users can directly build a PyTorch optimizer object and pass it to [Runner](https://mmengine.readthedocs.io/zh_CN/latest/tutorials/runner.html ). 
- Unlike the examples given in the PyTorch documentation, there is usually no need to manually implement a training loop and call ` optimizer.step()` in MMEngine; the executor automatically back-propagates the loss function and calls the optimizer's `step` method to update the model parameters.

- Also, we support building the optimizer from the registrar via a configuration file.
- Further, we provide optimizer constructor for more fine-grained tuning of the model optimization.



**(3) Usage of optimizers in mmengine**

The usage of the optimizer in mmengine can be found in the [documentation](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/optimizer.md)

**Basic usage:**

- Build the optimizer using the configuration file
- Setting up different **hyperparameters**
  - for different parameters in the model

- Set different **hyperparameter coefficients**
  - For different types of parameters
  - For parameters of different parts of the model

**Advanced usage:**

- Implementing a custom optimizer constructor

- Adjusting superparameters during training



### Implemented optimizers in MMSelfsup



- LARS

[LARS](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/mmselfsup/engine/optimizers/lars.py)

Implements layer-wise adaptive rate scaling for SGD.




- LearningRateDecayOptimWrapperConstructor

[LearningRateDecayOptimWrapperConstructor](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/mmselfsup/engine/optimizers/layer_decay_optim_wrapper_constructor.py#L64)

Different learning rates are set for different layers of backbone. Note: Currently, this optimizer constructor is built for ViT and Swin. In addition to applying layer-wise learning rate decay schedule, this module will not apply weight decay to ``normalization parameters``, ``bias``, ``position embedding``, ``class token``, and ``relative position bias table``, automatically. What's more, the ``paramwise_cfg`` in the base module will be ignored.



## Parameter Scheduler



### Introduction



**(1) Why use parameter scheduler**

During model training, we often do not use fixed optimization parameters, such as learning rate, which will be adjusted as the number of training rounds increases. The simplest and most common learning rate adjustment strategy is a step-down strategy, such as reducing the learning rate to a fraction of the original rate every once in a while.



**(2) What is the difference between the scheduler in mmengine and that in pytorch**

- In `mmengine.optim.scheduler`, we support most of the learning rate schedulers in PyTorch, such as `ExponentialLR`, `LinearLR`, `StepLR`, `MultiStepLR`, etc., and use them in the same way. interface documentation](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/TODO).

- We have also added the adjustment of momentum by replacing `LR` with `Momentum` in the class name, e.g. `ExponentialMomentum`, `LinearMomentum`.

- Further, we have implemented a generic parameter scheduler, ParamScheduler, to adjust other parameters in the optimizer, including weight_decay, etc. This feature makes it easy to configure complex tuning strategies for some of the new algorithms.

- Unlike the example given in the PyTorch documentation, the training loop and the call to `optimizer.step()` are usually not implemented manually in MMEngine, but the training process is managed automatically in the Runner, and the execution of the parameter scheduler is controlled by the `ParamSchedulerHook`.



**(3) Usage of the scheduler in mmengine**

The usage of the scheduler in mmengine can be found in [Parameter Scheduler Docs](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/param_scheduler .md)

The main topics include

- Using a single learning rate scheduler

- Combining multiple learning rate schedulers

- How to adjust other parameters (momentum, and generic parameter schedulers)





### Used Schedulers in MMSelfsup

MMSelfsup does not currently include a custom scheduling policy, and mainly uses the following three learning rate schedulers in mmengine, and their combinations.

- `CosineAnnealingLR`, defined in [here](https://github.com/open-mmlab/mmengine/blob/813f49bf23a5f454bca8ff01e65eca024825d447/mmengine/optim/ scheduler/lr_scheduler.py#L43)

- `LinearLR`, defined [here](https://github.com/open-mmlab/mmengine/blob/813f49bf23a5f454bca8ff01e65eca024825d447/mmengine/optim/scheduler/ lr_scheduler.py#L112)

- `MultiStepLR`, defined [here](https://github.com/open-mmlab/mmengine/blob/813f49bf23a5f454bca8ff01e65eca024825d447/mmengine/optim/ scheduler/lr_scheduler.py#L140)



### An example

For example, in configs/selfsup/mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py, the MAE scheduling policy is as follows:

```python
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=40,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=260,
        by_epoch=True,
        begin=40,
        end=300,
        convert_to_iter_based=True)
]
```

From epoch 0 to epoch 40, `LinearLR` is used; from epoch 40 to epoch 300, `CosineAnnealingLR` is used.