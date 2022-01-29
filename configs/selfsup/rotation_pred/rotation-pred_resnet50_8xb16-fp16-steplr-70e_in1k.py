_base_ = 'rotation-pred_resnet50_8xb16-steplr-70e_in1k.py'

# fp16
fp16 = dict(loss_scale=512.)
