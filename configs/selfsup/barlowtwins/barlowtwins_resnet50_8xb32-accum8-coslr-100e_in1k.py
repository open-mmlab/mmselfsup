_base_ = 'barlowtwins_resnet50_8xb256-coslr-300e_in1k.py'

data = dict(samples_per_gpu=32)

# additional hooks
# interval for accumulate gradient, total 8*32*8(interval)=2048
update_interval = 8

# optimizer
optimizer_config = dict(update_interval=update_interval)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
