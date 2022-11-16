_base_ = 'vit-base-p16_ft-8xb128-coslr-100e_in1k.py'

# model settings
model = dict(
    head=dict(init_cfg=[dict(type='TruncNormal', layer='Linear', std=0.02)
                        ]),  # MAE sets std to 2e-5
)

# optimizer wrapper
optim_wrapper = dict(
    optimizer=dict(lr=4e-4),  # layer-wise lr decay factor
)
