_base_ = 'deepcluster_resnrt50_8xb64-steplr-200e_in1k'

# model settings
model = dict(with_sobel=True, backbone=dict(in_channels=2))
