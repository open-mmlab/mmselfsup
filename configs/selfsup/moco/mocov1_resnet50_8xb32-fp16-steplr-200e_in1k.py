_base_ = 'mocov1_resnet50_8xb32-steplr-200e_in1k.py'

# fp16
fp16 = dict(loss_scale=512.)
