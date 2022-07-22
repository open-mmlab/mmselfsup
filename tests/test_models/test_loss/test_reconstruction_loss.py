# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.models import PixelReconstructionLoss


def test_reconstruction_loss():

    # test L2 loss
    loss_config = dict(criterion='L2')

    fake_pred = torch.rand((2, 196, 768))
    fake_target = torch.rand((2, 196, 768))
    fake_mask = torch.ones((2, 196))

    loss = PixelReconstructionLoss(**loss_config)
    loss_value = loss(fake_pred, fake_target, fake_mask)

    assert isinstance(loss_value.item(), float)

    # test L1 loss
    loss_config = dict(criterion='L1', channel=3)

    fake_pred = torch.rand((2, 3, 192, 192))
    fake_target = torch.rand((2, 3, 192, 192))
    fake_mask = torch.ones((2, 1, 192, 192))

    loss = PixelReconstructionLoss(**loss_config)
    loss_value = loss(fake_pred, fake_target, fake_mask)

    assert isinstance(loss_value.item(), float)
