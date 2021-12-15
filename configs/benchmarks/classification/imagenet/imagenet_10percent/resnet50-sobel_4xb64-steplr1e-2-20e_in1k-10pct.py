_base_ = 'resnet50_head1_4xb64-steplr1e-2-20e_in1k-10pct.py'

# model settings
model = dict(with_sobel=True, backbone=dict(in_channels=2))
