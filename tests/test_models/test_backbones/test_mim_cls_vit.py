# Copyright (c) OpenMMLab. All rights reserved.
import platform

import pytest
import torch

from mmselfsup.models.backbones import MIMVisionTransformer

finetune_backbone = dict(
    arch='b', patch_size=16, drop_path_rate=0.1, final_norm=False)

finetune_backbone_norm = dict(
    arch='b', patch_size=16, drop_path_rate=0.1, final_norm=True)

linprobe_backbone = dict(
    arch='b', patch_size=16, finetune=False, final_norm=False)


@pytest.mark.skipif(platform.system() == 'Windows', reason='Windows mem limit')
def test_mae_cls_vit():
    mae_finetune_backbone = MIMVisionTransformer(**finetune_backbone)
    mae_finetune_backbone_norm = MIMVisionTransformer(**finetune_backbone_norm)
    mae_linprobe_backbone = MIMVisionTransformer(**linprobe_backbone)
    mae_linprobe_backbone.train()

    fake_inputs = torch.randn((2, 3, 224, 224))
    fake_finetune_outputs = mae_finetune_backbone(fake_inputs)
    fake_finetune_outputs_norm = mae_finetune_backbone_norm(fake_inputs)
    fake_linprobe_outputs = mae_linprobe_backbone(fake_inputs)
    assert list(fake_finetune_outputs.shape) == [2, 768]
    assert list(fake_linprobe_outputs.shape) == [2, 768]
    assert list(fake_finetune_outputs_norm.shape) == [2, 768]
