_base_ = 'simsiam_resnet50_8xb32-coslr-200e_in1k.py'

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
