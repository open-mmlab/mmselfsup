# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmselfsup.models.backbones import ResNet
from mmselfsup.models.backbones.resnet import BasicBlock, Bottleneck


def is_block(modules):
    """Check if is ResNet building block."""
    if isinstance(modules, (BasicBlock, Bottleneck)):
        return True
    return False


def all_zeros(modules):
    """Check if the weight(and bias) is all zero."""
    weight_zero = torch.equal(modules.weight.data,
                              torch.zeros_like(modules.weight.data))
    if hasattr(modules, 'bias'):
        bias_zero = torch.equal(modules.bias.data,
                                torch.zeros_like(modules.bias.data))
    else:
        bias_zero = True

    return weight_zero and bias_zero


def check_norm_state(modules, train_state):
    """Check if norm layer is in correct train state."""
    for mod in modules:
        if isinstance(mod, _BatchNorm):
            if mod.training != train_state:
                return False
    return True


def test_resnet():
    """Test resnet backbone."""
    # Test ResNet50 norm_eval=True
    model = ResNet(50, norm_eval=True)
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test ResNet50 with torchvision pretrained weight
    model = ResNet(depth=50, norm_eval=True)
    model.init_weights()
    model.train()
    assert check_norm_state(model.modules(), False)

    # Test ResNet50 with first stage frozen
    frozen_stages = 1
    model = ResNet(50, frozen_stages=frozen_stages)
    model.init_weights()
    model.train()
    assert model.norm1.training is False
    for layer in [model.conv1, model.norm1]:
        for param in layer.parameters():
            assert param.requires_grad is False
    for i in range(1, frozen_stages + 1):
        layer = getattr(model, f'layer{i}')
        for mod in layer.modules():
            if isinstance(mod, _BatchNorm):
                assert mod.training is False
        for param in layer.parameters():
            assert param.requires_grad is False

    # Test ResNet18 forward
    model = ResNet(18, out_indices=(0, 1, 2, 3, 4))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 5
    assert feat[0].shape == (1, 64, 112, 112)
    assert feat[1].shape == (1, 64, 56, 56)
    assert feat[2].shape == (1, 128, 28, 28)
    assert feat[3].shape == (1, 256, 14, 14)
    assert feat[4].shape == (1, 512, 7, 7)

    # Test ResNet50 with BatchNorm forward
    model = ResNet(50, out_indices=(0, 1, 2, 3, 4))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 5
    assert feat[0].shape == (1, 64, 112, 112)
    assert feat[1].shape == (1, 256, 56, 56)
    assert feat[2].shape == (1, 512, 28, 28)
    assert feat[3].shape == (1, 1024, 14, 14)
    assert feat[4].shape == (1, 2048, 7, 7)

    # Test ResNet50 with layers 3 (top feature maps) out forward
    model = ResNet(50, out_indices=(4, ))
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert feat[0].shape == (1, 2048, 7, 7)

    # Test ResNet50 with checkpoint forward
    model = ResNet(50, out_indices=(0, 1, 2, 3, 4), with_cp=True)
    for m in model.modules():
        if is_block(m):
            assert m.with_cp
    model.init_weights()
    model.train()

    imgs = torch.randn(1, 3, 224, 224)
    feat = model(imgs)
    assert len(feat) == 5
    assert feat[0].shape == (1, 64, 112, 112)
    assert feat[1].shape == (1, 256, 56, 56)
    assert feat[2].shape == (1, 512, 28, 28)
    assert feat[3].shape == (1, 1024, 14, 14)
    assert feat[4].shape == (1, 2048, 7, 7)

    # zero initialization of residual blocks
    model = ResNet(50, zero_init_residual=True)
    model.init_weights()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            assert all_zeros(m.norm3)
        elif isinstance(m, BasicBlock):
            assert all_zeros(m.norm2)

    # non-zero initialization of residual blocks
    model = ResNet(50, zero_init_residual=False)
    model.init_weights()
    for m in model.modules():
        if isinstance(m, Bottleneck):
            assert not all_zeros(m.norm3)
        elif isinstance(m, BasicBlock):
            assert not all_zeros(m.norm2)
