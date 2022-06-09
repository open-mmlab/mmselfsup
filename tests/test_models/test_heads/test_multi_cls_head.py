# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
from mmcls.core.data_structures import ClsDataSample

from mmselfsup.models.heads import MultiClsHead


class TestMultiClsHead(TestCase):

    def test_loss(self):
        head = MultiClsHead(in_indices=(0, 1))
        fake_in = [torch.rand(2, 64, 112, 112), torch.rand(2, 256, 56, 56)]
        fake_out = head.forward(fake_in)
        assert isinstance(fake_out, list)

        fake_data_samples = [ClsDataSample().set_gt_label(1) for _ in range(2)]
        losses = head.loss(fake_in, fake_data_samples)
        print(losses)
        self.assertEqual(len(losses.keys()), 4)
        for k in losses.keys():
            assert k.startswith('loss') or k.startswith('accuracy')
            if k.startswith('loss'):
                self.assertGreater(losses[k].item(), 0)

    def test_predict(self):
        head = MultiClsHead(in_indices=(0, 1))
        fake_in = [torch.rand(2, 64, 112, 112), torch.rand(2, 256, 56, 56)]
        fake_data_samples = [ClsDataSample().set_gt_label(1) for _ in range(2)]

        fake_out = head.predict(fake_in, fake_data_samples)
        self.assertEqual(len(fake_out), 2)
        for i in range(2):
            self.assertEqual(fake_out[i].head0_pred_label.score.size(),
                             (1000, ))
            self.assertEqual(fake_out[i].head1_pred_label.score.size(),
                             (1000, ))
