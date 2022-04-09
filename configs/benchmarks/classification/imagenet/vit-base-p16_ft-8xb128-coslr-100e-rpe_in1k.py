_base_ = 'vit-b-p16_ft-8xb128-coslr-100e_in1k.py'

# model
model = dict(backbone=dict(use_window=True, init_values=0.1))

# optimizer
optimizer = dict(lr=8e-3)

# learning policy
lr_config = dict(warmup_iters=10)

find_unused_parameters = True
