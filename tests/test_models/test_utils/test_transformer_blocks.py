# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import torch
import torch.nn as nn

from mmselfsup.models.utils.transformer_blocks import (
    CAETransformerRegressorLayer, CrossMultiheadAttention, MultiheadAttention,
    MultiheadAttentionWithRPE, TransformerEncoderLayer)


class TestTransformerBlocks(TestCase):

    def test_multihead_attention(self):
        module = MultiheadAttention(4, 2, qkv_bias=False)
        assert module.q_bias.data.equal(torch.zeros(4))
        assert module.v_bias.data.equal(torch.zeros(4))

        fake_input = torch.rand((2, 2, 4))
        fake_output = module(fake_input)
        self.assertEqual(fake_input.size(), fake_output.size())

    def test_multihead_attention_with_rpe(self):
        # qkv_bias is True
        module = MultiheadAttentionWithRPE(4, 2, (1, 1))
        assert module.q_bias.data.equal(torch.zeros(4))
        assert module.v_bias.data.equal(torch.zeros(4))

        fake_input = torch.rand((2, 2, 4))
        fake_output = module(fake_input)
        self.assertEqual(fake_input.size(), fake_output.size())

        # qkv_bias is False
        module = MultiheadAttentionWithRPE(4, 2, (1, 1), qkv_bias=False)
        assert module.q_bias is None
        assert module.k_bias is None
        assert module.v_bias is None

        fake_input = torch.rand((2, 2, 4))
        fake_output = module(fake_input)
        self.assertEqual(fake_input.size(), fake_output.size())

    def test_transformer_encoder_layer(self):
        # test init_values
        module = TransformerEncoderLayer(4, 2, 2, init_values=0.5)
        assert module.gamma_1.data.equal(0.5 * torch.ones((4)))
        assert module.gamma_2.data.equal(0.5 * torch.ones((4)))

        # test without rpe
        module = TransformerEncoderLayer(4, 2, 2)
        assert module.gamma_1 is None
        assert module.gamma_2 is None

        fake_input = torch.rand((2, 2, 4))
        fake_output = module(fake_input)
        self.assertEqual(fake_input.size(), fake_output.size())

        # test with rpe
        module = TransformerEncoderLayer(4, 2, 2, window_size=(1, 1))
        assert module.gamma_1 is None
        assert module.gamma_2 is None

        fake_input = torch.rand((2, 2, 4))
        fake_output = module(fake_input)
        self.assertEqual(fake_input.size(), fake_output.size())

    def test_cae_transformer_regression_layer(self):
        # test init_values
        module = CAETransformerRegressorLayer(4, 2, 2)
        assert module.gamma_1_cross.equal(
            nn.Parameter(torch.ones((4)), requires_grad=False))
        assert module.gamma_2_cross.equal(
            nn.Parameter(torch.ones((4)), requires_grad=False))

        module = CAETransformerRegressorLayer(4, 2, 2, init_values=0.5)
        assert module.gamma_1_cross.data.equal(0.5 * torch.ones((4)))
        assert module.gamma_2_cross.data.equal(0.5 * torch.ones((4)))

    def test_cross_multihead_attention(self):
        # qkv_bias is True
        module = CrossMultiheadAttention(4, 2, qkv_bias=True)
        assert module.q_bias.data.equal(torch.zeros(4))
        assert module.v_bias.data.equal(torch.zeros(4))

        # qkv_bias is False
        module = CrossMultiheadAttention(4, 2, qkv_bias=False)
        assert module.q_bias is None
        assert module.k_bias is None
        assert module.v_bias is None
