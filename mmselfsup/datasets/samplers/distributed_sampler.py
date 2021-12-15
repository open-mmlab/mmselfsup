# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import Sampler


class DistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 replace=False):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.replace = replace
        self.unif_sampling_flag = False

    def __iter__(self):
        # deterministically shuffle based on epoch
        if not self.unif_sampling_flag:
            self.generate_new_list()
        else:
            self.unif_sampling_flag = False
        return iter(self.indices[self.rank * self.num_samples:(self.rank + 1) *
                                 self.num_samples])

    def generate_new_list(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
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
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        self.indices = indices

    def set_uniform_indices(self, labels, num_classes):
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
        indices = indices[:N].astype(np.int).tolist()

        # add extra samples to make it evenly divisible
        assert len(indices) <= self.total_size, \
            f'{len(indices)} vs {self.total_size}'
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size, \
            f'{len(indices)} vs {self.total_size}'
        self.indices = indices


class DistributedGivenIterationSampler(Sampler):

    def __init__(self,
                 dataset,
                 total_iter,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 last_iter=-1):
        rank, world_size = get_dist_info()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter * self.batch_size

        self.indices = self.gen_new_list()

    def __iter__(self):
        return iter(self.indices[(self.last_iter + 1) * self.batch_size:])

    def set_uniform_indices(self, labels, num_classes):
        np.random.seed(0)
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
        indices = indices[:N].astype(np.int)
        # repeat
        all_size = self.total_size * self.world_size
        indices = indices[:all_size]
        num_repeat = (all_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]
        np.random.shuffle(indices)
        # slice
        beg = self.total_size * self.rank
        indices = indices[beg:beg + self.total_size]
        assert len(indices) == self.total_size
        # set
        self.indices = indices

    def gen_new_list(self):
        """Each process shuffle all list with same seed, and pick one piece
        according to rank."""
        np.random.seed(0)

        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        beg = self.total_size * self.rank
        indices = indices[beg:beg + self.total_size]

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        """Note here we do not take last iter into consideration, since __len__
        should only be used for displaying, the correct remaining size is
        handled by dataloader."""

        # return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size

    def set_epoch(self, epoch):
        pass
