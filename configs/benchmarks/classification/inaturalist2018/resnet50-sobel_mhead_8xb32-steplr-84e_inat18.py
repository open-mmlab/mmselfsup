_base_ = 'resnet50_mhead_8xb32-steplr-84e_inat18.py'

# model settings
model = dict(with_sobel=True, backbone=dict(in_channels=2))
