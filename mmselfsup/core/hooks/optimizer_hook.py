# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import (HOOKS, Fp16OptimizerHook, OptimizerHook,
                         allreduce_grads)
from mmcv.utils import TORCH_VERSION, _BatchNorm, digit_version


@HOOKS.register_module()
class DistOptimizerHook(OptimizerHook):
    """Optimizer hook for distributed training.

    This hook can accumulate gradients every n intervals and freeze some
    layers for some iters at the beginning.

    Args:
        update_interval (int, optional): The update interval of the weights,
            set > 1 to accumulate the grad. Defaults to 1.
        grad_clip (dict, optional): Dict to config the value of grad clip.
            E.g., grad_clip = dict(max_norm=10). Defaults to None.
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
        frozen_layers_cfg (dict, optional): Dict to config frozen layers.
            The key-value pair is layer name and its frozen iters. If frozen,
            the layer gradient would be set to None. Defaults to dict().
    """

    def __init__(self,
                 update_interval=1,
                 grad_clip=None,
                 coalesce=True,
                 bucket_size_mb=-1,
                 frozen_layers_cfg=dict()):
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.update_interval = update_interval
        self.frozen_layers_cfg = frozen_layers_cfg
        self.initialized = False

    def has_batch_norm(self, module):
        if isinstance(module, _BatchNorm):
            return True
        for m in module.children():
            if self.has_batch_norm(m):
                return True
        return False

    def _init(self, runner):
        if runner.iter % self.update_interval != 0:
            runner.logger.warning(
                'Resume iter number is not divisible by update_interval in '
                'GradientCumulativeOptimizerHook, which means the gradient of '
                'some iters is lost and the result may be influenced slightly.'
            )

        if self.has_batch_norm(runner.model) and self.update_interval > 1:
            runner.logger.warning(
                'GradientCumulativeOptimizerHook may slightly decrease '
                'performance if the model has BatchNorm layers.')

        residual_iters = runner.max_iters

        self.divisible_iters = (
            residual_iters // self.update_interval * self.update_interval)
        self.remainder_iters = residual_iters - self.divisible_iters

        self.initialized = True

    def before_run(self, runner):
        runner.optimizer.zero_grad()

    def after_train_iter(self, runner):
        # In some cases, MMCV's GradientCumulativeOptimizerHook will
        # cause the loss_factor to be zero and we fix this bug in our
        # implementation.

        if not self.initialized:
            self._init(runner)

        if runner.iter < self.divisible_iters:
            loss_factor = self.update_interval
        else:
            loss_factor = self.remainder_iters

        runner.outputs['loss'] /= loss_factor
        runner.outputs['loss'].backward()

        if (self.every_n_iters(runner, self.update_interval)
                or self.is_last_iter(runner)):

            # cancel gradient of certain layer for n iters
            # according to frozen_layers_cfg dict
            for layer, iters in self.frozen_layers_cfg.items():
                if runner.iter < iters:
                    for name, p in runner.model.module.named_parameters():
                        if layer in name:
                            p.grad = None

            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Add grad norm to the logger
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                             runner.outputs['num_samples'])

            runner.optimizer.step()
            runner.optimizer.zero_grad()


if (TORCH_VERSION != 'parrots'
        and digit_version(TORCH_VERSION) >= digit_version('1.6.0')):

    @HOOKS.register_module()
    class GradAccumFp16OptimizerHook(Fp16OptimizerHook):
        """Fp16 optimizer hook (using PyTorch's implementation).

        This hook can accumulate gradients every n intervals and freeze some
        layers for some iters at the beginning.
        If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
        to take care of the optimization procedure.

        Args:
            update_interval (int, optional): The update interval of the
                weights, set > 1 to accumulate the grad. Defaults to 1.
            frozen_layers_cfg (dict, optional): Dict to config frozen layers.
                The key-value pair is layer name and its frozen iters. If
                frozen, the layer gradient would be set to None.
                Defaults to dict().
        """

        def __init__(self,
                     update_interval=1,
                     frozen_layers_cfg=dict(),
                     **kwargs):
            super(GradAccumFp16OptimizerHook, self).__init__(**kwargs)
            self.update_interval = update_interval
            self.frozen_layers_cfg = frozen_layers_cfg

        def after_train_iter(self, runner):
            runner.outputs['loss'] /= self.update_interval
            self.loss_scaler.scale(runner.outputs['loss']).backward()

            if self.every_n_iters(runner, self.update_interval):

                # cancel gradient of certain layer for n iters
                # according to frozen_layers_cfg dict
                for layer, iters in self.frozen_layers_cfg.items():
                    if runner.iter < iters:
                        for name, p in runner.model.module.named_parameters():
                            if layer in name:
                                p.grad = None

                # copy fp16 grads in the model to fp32 params in the optimizer
                self.loss_scaler.unscale_(runner.optimizer)

                if self.grad_clip is not None:
                    grad_norm = self.clip_grads(runner.model.parameters())
                    if grad_norm is not None:
                        # Add grad norm to the logger
                        runner.log_buffer.update(
                            {'grad_norm': float(grad_norm)},
                            runner.outputs['num_samples'])

                # backward and update scaler
                self.loss_scaler.step(runner.optimizer)
                self.loss_scaler.update(self._scale_update_param)

                # save state_dict of loss_scaler
                runner.meta.setdefault(
                    'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

                # clear grads
                runner.model.zero_grad()
                runner.optimizer.zero_grad()

else:

    @HOOKS.register_module()
    class GradAccumFp16OptimizerHook(Fp16OptimizerHook):
        """Fp16 optimizer hook (using mmcv's implementation).

        This hook can accumulate gradients every n intervals and freeze some
        layers for some iters at the beginning.

        Args:
            update_interval (int, optional): The update interval of the
                weights, set > 1 to accumulate the grad. Defaults to 1.
            frozen_layers_cfg (dict, optional): Dict to config frozen layers.
                The key-value pair is layer name and its frozen iters. If
                frozen, the layer gradient would be set to None.
                Defaults to dict().
        """

        def __init__(self,
                     update_interval=1,
                     frozen_layers_cfg=dict(),
                     **kwargs):
            super(GradAccumFp16OptimizerHook, self).__init__(**kwargs)
            self.update_interval = update_interval
            self.frozen_layers_cfg = frozen_layers_cfg

        def after_train_iter(self, runner):
            runner.outputs['loss'] /= self.update_interval

            # scale the loss value
            scaled_loss = runner.outputs['loss'] * self.loss_scaler.loss_scale
            scaled_loss.backward()

            if self.every_n_iters(runner, self.update_interval):

                # cancel gradient of certain layer for n iters
                # according to frozen_layers_cfg dict
                for layer, iters in self.frozen_layers_cfg.items():
                    if runner.iter < iters:
                        for name, p in runner.model.module.named_parameters():
                            if layer in name:
                                p.grad = None

                # copy fp16 grads in the model to fp32 params in the optimizer
                fp32_weights = []
                for param_group in runner.optimizer.param_groups:
                    fp32_weights += param_group['params']
                self.copy_grads_to_fp32(runner.model, fp32_weights)
                # allreduce grads
                if self.distributed:
                    allreduce_grads(fp32_weights, self.coalesce,
                                    self.bucket_size_mb)

                has_overflow = self.loss_scaler.has_overflow(fp32_weights)
                # if has overflow, skip this iteration
                if not has_overflow:
                    # scale the gradients back
                    for param in fp32_weights:
                        if param.grad is not None:
                            param.grad.div_(self.loss_scaler.loss_scale)
                    if self.grad_clip is not None:
                        grad_norm = self.clip_grads(fp32_weights)
                        if grad_norm is not None:
                            # Add grad norm to the logger
                            runner.log_buffer.update(
                                {'grad_norm': float(grad_norm)},
                                runner.outputs['num_samples'])
                    # update fp32 params
                    runner.optimizer.step()
                    # copy fp32 params to the fp16 model
                    self.copy_params_to_fp16(runner.model, fp32_weights)
                else:
                    runner.logger.warning(
                        'Check overflow, downscale loss scale '
                        f'to {self.loss_scaler.cur_scale}')

                self.loss_scaler.update_scale(has_overflow)

                # save state_dict of loss_scaler
                runner.meta.setdefault(
                    'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

                # clear grads
                runner.model.zero_grad()
                runner.optimizer.zero_grad()
