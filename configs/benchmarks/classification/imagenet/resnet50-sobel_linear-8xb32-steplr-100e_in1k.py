_base_ = 'resnet50_linear-8xb32-steplr-100e_in1k.py'

# model settings
model = dict(with_sobel=True, backbone=dict(in_channels=2, frozen_stages=4))
