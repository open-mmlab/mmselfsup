_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/datasets/cifar10.py',
    '../_base_/schedules/sgd_steplr-100e.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(head=dict(num_classes=10))

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=5e-4)

# learning policy
lr_config = dict(policy='step', step=[150, 250])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=350)
checkpoint_config = dict(interval=50)
