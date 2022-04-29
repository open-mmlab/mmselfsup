_base_ = 'barlowtwins_resnet50_8xb256-coslr-300e_in1k.py'

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1000)
