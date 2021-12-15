_base_ = 'npid_resnet50_8xb32-steplr-200e_in1k.py'

# model settings
model = dict(ensure_neg=True)
