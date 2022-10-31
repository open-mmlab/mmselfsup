_base_ = 'resnet50_head1_4xb64-steplr1e-1-20e_in1k-1pct.py'

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
