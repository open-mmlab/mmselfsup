_base_ = 'deepcluster_resnet50_8xb64-steplr-200e_in1k.py'

# model settings
model = dict(with_sobel=True, backbone=dict(in_channels=2))
