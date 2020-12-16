from mmcv.runner import Hook

import torch
from torch.utils.data import Dataset

from openselfsup.utils import nondist_forward_collect, dist_forward_collect
from .registry import HOOKS


@HOOKS.register_module
class ValidateHook(Hook):
    """Validation hook.

    Args:
        dataset (Dataset | dict): A PyTorch dataset or dict that indicates
            the dataset.
        dist_mode (bool): Use distributed evaluation or not. Default: True.
        initial (bool): Whether to evaluate before the training starts.
            Default: True.
        interval (int): Evaluation interval (by epochs). Default: 1.
        **eval_kwargs: Evaluation arguments fed into the evaluate function of
            the dataset.
    """

    def __init__(self,
                 dataset,
                 dist_mode=True,
                 initial=True,
                 interval=1,
                 **eval_kwargs):
        from openselfsup import datasets
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset)
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.data_loader = datasets.build_dataloader(
            self.dataset,
            eval_kwargs['imgs_per_gpu'],
            eval_kwargs['workers_per_gpu'],
            dist=dist_mode,
            shuffle=False,
            prefetch=eval_kwargs.get('prefetch', False),
            img_norm_cfg=eval_kwargs.get('img_norm_cfg', dict()),
        )
        self.dist_mode = dist_mode
        self.initial = initial
        self.interval = interval
        self.eval_kwargs = eval_kwargs

    def before_run(self, runner):
        if self.initial:
            self._run_validate(runner)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        self._run_validate(runner)

    def _run_validate(self, runner):
        runner.model.eval()
        func = lambda **x: runner.model(mode='test', **x)
        if self.dist_mode:
            results = dist_forward_collect(
                func, self.data_loader, runner.rank,
                len(self.dataset))  # dict{key: np.ndarray}
        else:
            results = nondist_forward_collect(func, self.data_loader,
                                              len(self.dataset))
        if runner.rank == 0:
            for name, val in results.items():
                self._evaluate(runner, torch.from_numpy(val), name)
        runner.model.train()

    def _evaluate(self, runner, results, keyword):
        eval_res = self.dataset.evaluate(
            results,
            keyword=keyword,
            logger=runner.logger,
            **self.eval_kwargs['eval_param'])
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True
