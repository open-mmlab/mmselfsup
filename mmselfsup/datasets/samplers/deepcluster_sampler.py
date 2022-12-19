# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterator, Optional, Sized

import numpy as np
import torch
from mmengine.dataset import DefaultSampler

from mmselfsup.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class DeepClusterSampler(DefaultSampler):
    """The sampler inherits ``DefaultSampler`` from mmengine.

    This sampler supports to set replace to be ``True`` to get indices.
    Besides, it defines function ``set_uniform_indices``, which is applied in
    ``DeepClusterHook``.

    Args:
        dataset (Sized): The dataset.
        shuffle (bool): Whether shuffle the dataset or not. Defaults to True.
        seed (int, optional): Random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Defaults to None.
        replace (bool): Replace or not in random shuffle.
            It works on when shuffle is True. Defaults to False.
        round_up (bool): Whether to add extra samples to make the number of
            samples evenly divisible by the world size. Defaults to True.
    """

    def __init__(self,
                 dataset: Sized,
                 shuffle: bool = True,
                 seed: Optional[int] = None,
                 replace: bool = False,
                 round_up: bool = True) -> None:
        super().__init__(
            dataset=dataset, shuffle=shuffle, seed=seed, round_up=round_up)
        self.replace = replace
        self.unif_sampling_flag = False

    def __iter__(self) -> Iterator[int]:
        """Iterate the indices."""
        # deterministically shuffle based on epoch and seed
        if not self.unif_sampling_flag:
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                if self.replace:
                    indices = torch.randint(
                        low=0,
                        high=len(self.dataset),
                        size=(len(self.dataset), ),
                        generator=g).tolist()
                else:
                    indices = torch.randperm(
                        len(self.dataset), generator=g).tolist()
            else:
                indices = torch.arange(len(self.dataset)).tolist()

            # add extra samples to make it evenly divisible
            indices = (
                indices *
                int(self.total_size / len(indices) + 1))[:self.total_size]
            assert len(indices) == self.total_size
            self.indices = indices
        else:
            self.unif_sampling_flag = False

        # subsample
        self.indices = self.indices[self.rank:self.total_size:self.world_size]

        return iter(self.indices)

    def set_uniform_indices(self, labels: list, num_classes: int) -> None:
        """The function is applied in DeepClusterHook for uniform sampling.

        Args:
            labels (list): The updated labels after clustering.
            num_classes (int): number of clusters.

        Returns:
            None
        """
        self.unif_sampling_flag = True
        assert self.shuffle,\
            'Using uniform sampling, the indices must be shuffled.'
        np.random.seed(self.epoch)
        assert (len(labels) == len(self.dataset))
        N = len(labels)
        size_per_label = int(N / num_classes) + 1
        indices = []
        images_lists = [[] for i in range(num_classes)]
        for i, l in enumerate(labels):
            images_lists[l].append(i)
        for i, l in enumerate(images_lists):
            if len(l) == 0:
                continue
            indices.extend(
                np.random.choice(
                    l, size_per_label, replace=(len(l) <= size_per_label)))
        indices = np.array(indices)
        np.random.shuffle(indices)
        indices = indices[:N].astype(int).tolist()

        # add extra samples to make it evenly divisible
        assert len(indices) <= self.total_size, \
            f'{len(indices)} vs {self.total_size}'
        indices = (indices *
                   int(self.total_size / len(indices) + 1))[:self.total_size]
        assert len(indices) == self.total_size, \
            f'{len(indices)} vs {self.total_size}'
        self.indices = indices
