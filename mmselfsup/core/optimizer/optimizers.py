# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner.optimizer.builder import OPTIMIZERS
from torch.optim import *  # noqa: F401,F403
from torch.optim.optimizer import Optimizer, required


@OPTIMIZERS.register_module()
class LARS(Optimizer):
    """Implements layer-wise adaptive rate scaling for SGD.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Base learning rate.
        momentum (float, optional): Momentum factor. Defaults to 0 ('m')
        weight_decay (float, optional): Weight decay (L2 penalty).
            Defaults to 0. ('beta')
        dampening (float, optional): Dampening for momentum. Defaults to 0.
        eta (float, optional): LARS coefficient. Defaults to 0.001.
        nesterov (bool, optional): Enables Nesterov momentum.
            Defaults to False.
        eps (float, optional): A small number to avoid dviding zero.
            Defaults to 1e-8.

    Based on Algorithm 1 of the following paper by You, Gitman, and Ginsburg.
    `Large Batch Training of Convolutional Networks:
        <https://arxiv.org/abs/1708.03888>`_.

    Example:
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9,
        >>>                  weight_decay=1e-4, eta=1e-3)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    """

    def __init__(self,
                 params,
                 lr=required,
                 momentum=0,
                 weight_decay=0,
                 dampening=0,
                 eta=0.001,
                 nesterov=False,
                 eps=1e-8):
        if lr is not required and lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if momentum < 0.0:
            raise ValueError(f'Invalid momentum value: {momentum}')
        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')
        if eta < 0.0:
            raise ValueError(f'Invalid LARS coefficient value: {eta}')

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            eta=eta)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                'Nesterov momentum requires a momentum and zero dampening')

        self.eps = eps
        super(LARS, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(LARS, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            eta = group['eta']
            nesterov = group['nesterov']
            lr = group['lr']
            lars_exclude = group.get('lars_exclude', False)

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad

                if lars_exclude:
                    local_lr = 1.
                else:
                    weight_norm = torch.norm(p).item()
                    grad_norm = torch.norm(d_p).item()
                    if weight_norm != 0 and grad_norm != 0:
                        # Compute local learning rate for this layer
                        local_lr = eta * weight_norm / \
                            (grad_norm + weight_decay * weight_norm + self.eps)
                    else:
                        local_lr = 1.

                actual_lr = local_lr * lr
                d_p = d_p.add(p, alpha=weight_decay).mul(actual_lr)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = \
                                torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                p.add_(-d_p)

        return loss
