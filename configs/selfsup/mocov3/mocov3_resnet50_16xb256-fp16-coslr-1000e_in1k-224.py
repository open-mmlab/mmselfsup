_base_ = 'mocov3_resnet50_16xb256-fp16-coslr-100e_in1k-224.py'

model = dict(base_momentum=0.996)  # 0.99 for 100e and 300e, 0.996 for 1000e

# optimizer
optimizer = dict(type='LARS', lr=4.8, weight_decay=1.5e-6)
optim_wrapper = dict(optimizer=optimizer)

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=990,
        by_epoch=True,
        begin=10,
        end=1000,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1000)
