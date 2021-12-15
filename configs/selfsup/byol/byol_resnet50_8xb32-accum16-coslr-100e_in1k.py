_base_ = 'byol_resnet50_8xb32-accum16-coslr-200e_in1k.py'

# optimizer
optimizer = dict(lr=7.2)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
