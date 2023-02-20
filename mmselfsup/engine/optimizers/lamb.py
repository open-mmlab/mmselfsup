# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer

from mmselfsup.registry import OPTIMIZERS


@OPTIMIZERS.register_module()
class LAMB(Optimizer):
    """LAMB optimizer implementation,
    `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes:
    <https://arxiv.org/abs/1904.00962>`_.

    The codes are modified from:
    https://github.com/keyu-tian/SparK/blob/main/utils/optim.py

    Arguments:
        params (Iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Learning rate.
        betas (Tuple[float, float], optional): Coefficients used for computing
            running averages of gradient and its norm. Defaults to (0.9, 0.999)
        eps (float, optional): Term added to the denominator to improve
            numerical stability. Defaults to 1e-8.
        weight_decay (float, optional): Weight decay (L2 penalty).
            Defaults to 0.
        grad_averaging (bool, optional): Whether apply (1-beta2) to grad when
            calculating running averages of gradient. Defaults to True.
        trust_clip (bool, optional): Enable LAMBC trust ratio clipping.
            Defaults to False.
        always_adapt (bool, optional): Apply adaptive learning rate to 0.0
            weight decay parameter. Defaults to False.
    """

    def __init__(self,
                 params: Iterable,
                 lr: float,
                 bias_correction: bool = True,
                 betas: Optional[tuple] = (0.9, 0.999),
                 eps: float = 1e-6,
                 weight_decay: float = 0.01,
                 grad_averaging: bool = True,
                 trust_clip: bool = False,
                 always_adapt: bool = False):
        defaults = dict(
            lr=lr,
            bias_correction=bias_correction,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_averaging=grad_averaging,
            trust_clip=trust_clip,
            always_adapt=always_adapt)
        super().__init__(params, defaults)
        self.global_grad_norm = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = self.param_groups[0]['params'][0].device
        # because torch.where doesn't handle scalars correctly
        one_tensor = torch.tensor(1.0, device=device)
        global_grad_norm = torch.zeros(1, device=device)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        'Lamb does not support sparse gradients, consider '
                        'SparseAdam instad.')
                global_grad_norm.add_(grad.pow(2).sum())

        global_grad_norm = torch.sqrt(global_grad_norm)
        self.global_grad_norm = global_grad_norm.item()

        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0
            beta3 = 1 - beta1 if grad_averaging else 1.0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor,
            # or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            if bias_correction:
                bias_correction1 = 1 - beta1**group['step']
                bias_correction2 = 1 - beta2**group['step']
            else:
                bias_correction1, bias_correction2 = 1.0, 1.0

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.mul_(global_grad_norm)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient valuesa
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=beta3)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=1 - beta2)  # v_t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group['eps'])
                update = (exp_avg / bias_correction1).div_(denom)

                weight_decay = group['weight_decay']
                if weight_decay != 0:
                    update.add_(p, alpha=weight_decay)

                if weight_decay != 0 or group['always_adapt']:
                    # Layer-wise LR adaptation.
                    # By default, skip adaptation on parameters that are
                    # excluded from weight decay, unless always_adapt == True,
                    # then always enabled.
                    w_norm = p.norm(2.0)
                    g_norm = update.norm(2.0)
                    # FIXME nested where required since logical and/or not
                    # working in PT XLA
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, w_norm / g_norm, one_tensor),
                        one_tensor,
                    )
                    if group['trust_clip']:
                        # LAMBC trust clipping, upper bound fixed at one
                        trust_ratio = torch.minimum(trust_ratio, one_tensor)
                    update.mul_(trust_ratio)

                p.add_(update, alpha=-group['lr'])

        return loss
