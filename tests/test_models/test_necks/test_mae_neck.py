# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmselfsup.models.necks import MAEPretrainDecoder


def test_linear_neck():
    decoder = MAEPretrainDecoder()
    decoder.init_weights()
    decoder.eval()
    inputs = torch.rand(1, 50, 1024)
    ids_restore = torch.arange(0, 196).unsqueeze(0)
    level_outputs = decoder.forward(inputs, ids_restore)
    assert tuple(level_outputs.shape) == (1, 196, 768)
