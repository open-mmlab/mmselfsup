# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.utils import build_from_cfg, print_log
from torchvision.transforms import Compose

from .base import BaseDataset
from .builder import DATASETS, PIPELINES, build_datasource


@DATASETS.register_module()
class MAEFtDataset(BaseDataset):
    """The dataset outputs the augmented image and the mask.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        pipeline (list[[dict]]): A list of data augmentations,
            where each augmentaion contains element that represents
            an operation defined in `mmselfsup.datasets.pipelines.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.

    Examples:
        >>> dataset = MAEDataset(data_source, [pipeline])
        >>> img, mask = dataset[idx]
        The dataset will return the augmented image and the mask
    """

    def __init__(self, data_source, pipeline, prefetch=False):
        self.data_source = build_datasource(data_source)
        self.trans = Compose([build_from_cfg(p, PIPELINES) for p in pipeline])
        self.gt_labels = self.data_source.get_gt_labels()

    def __getitem__(self, idx):
        img = self.data_source.get_img(idx)
        label = self.gt_labels[idx]
        img = self.trans(img)

        return dict(img=img, label=label)

    def evaluate(self, results, logger=None, topk=(1, 5)):
        """The evaluation function to output accuracy.

        Args:
            results (dict): The key-value pair is the output head name and
                corresponding prediction values.
            logger (logging.Logger | str | None, optional): The defined logger
                to be used. Defaults to None.
            topk (tuple(int)): The output includes topk accuracy.
        """
        eval_res = {}
        for name, val in results.items():
            val = torch.from_numpy(val)
            target = torch.LongTensor(self.data_source.get_gt_labels())
            assert val.size(0) == target.size(0), (
                f'Inconsistent length for results and labels, '
                f'{val.size(0)} vs {target.size(0)}')

            num = val.size(0)
            _, pred = val.topk(max(topk), dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))  # [K, N]
            for k in topk:
                correct_k = correct[:k].contiguous().view(-1).float().sum(
                    0).item()
                acc = correct_k * 100.0 / num
                eval_res[f'{name}_top{k}'] = acc
                if logger is not None and logger != 'silent':
                    print_log(f'{name}_top{k}: {acc:.03f}', logger=logger)
        return eval_res
