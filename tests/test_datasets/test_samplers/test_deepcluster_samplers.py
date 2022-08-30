# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase
from unittest.mock import patch

from mmengine.dataset.sampler import DefaultSampler

from mmselfsup.datasets.samplers import DeepClusterSampler


class TestDeepClusterSampler(TestCase):

    def setUp(self):
        self.data_length = 100
        self.dataset = list(range(self.data_length))

    @patch('mmengine.dataset.sampler.get_dist_info', return_value=(0, 1))
    def test_deepcluster_sampler(self, mock):
        # test round_up=True
        sampler = DeepClusterSampler(
            self.dataset, round_up=True, shuffle=False)
        self.assertEqual(sampler.total_size, self.data_length)
        self.assertEqual(sampler.num_samples, self.data_length)
        self.assertEqual(list(sampler), list(range(self.data_length)))

        # test round_up=False
        sampler = DeepClusterSampler(
            self.dataset, round_up=False, shuffle=False)
        self.assertEqual(sampler.total_size, self.data_length)
        self.assertEqual(sampler.num_samples, self.data_length)
        self.assertEqual(list(sampler), list(range(self.data_length)))

        # test the consistency
        default_sampler = DefaultSampler(self.dataset, seed=0)
        sampler = DeepClusterSampler(self.dataset, seed=0, replace=False)
        self.assertEqual(list(sampler), list(default_sampler))

        # test replace
        default_sampler = DefaultSampler(self.dataset, seed=0)
        sampler = DeepClusterSampler(self.dataset, seed=0, replace=True)
        self.assertEqual(len(list(sampler)), self.data_length)
        self.assertNotEqual(list(sampler), list(default_sampler))

    @patch('mmengine.dataset.sampler.get_dist_info', return_value=(0, 1))
    def test_set_uniform_indices(self, mock):
        sampler = DeepClusterSampler(self.dataset, seed=0, replace=True)
        idx_0 = list(sampler)

        labels = [0] * 50 + [1] * 50
        sampler.set_uniform_indices(labels, 2)
        idx_1 = list(sampler)

        self.assertEqual(len(idx_0), len(idx_1))
        self.assertNotEqual(idx_0, idx_1)
