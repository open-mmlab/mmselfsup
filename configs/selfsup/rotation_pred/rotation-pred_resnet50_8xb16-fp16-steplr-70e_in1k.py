_base_ = 'rotation-pred_resnet50_8xb16-steplr-70e_in1k.py'

# mixed precision
optim_wrapper = dict(type='AmpOptimWrapper')
