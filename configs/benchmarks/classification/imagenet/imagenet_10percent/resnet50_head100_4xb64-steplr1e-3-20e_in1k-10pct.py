_base_ = 'resnet50_head1_4xb64-steplr1e-1-20e_in1k-10pct.py'

# optimizer
optimizer = dict(lr=0.001)
optim_wrapper = dict(
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=100.)}))
