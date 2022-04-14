_base_ = 'vit-base-p16_ft-8xb128-coslr-100e-rpe_in1k.py'

# learning policy
lr_config = dict(warmup_iters=5)