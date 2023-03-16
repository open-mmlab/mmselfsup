_base_ = '../../../benchmarks/classification/imagenet/vit-base-p16_linear-8xb2048-coslr-90e_in1k.py'  # noqa: E501

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmcls.ToPIL', to_rgb=True),
    dict(
        type='mmcls.torchvision/RandomResizedCrop', size=224, interpolation=3),
    dict(type='mmcls.torchvision/RandomHorizontalFlip', p=0.5),
    dict(type='mmcls.ToNumpy', to_rgb=True),
    dict(type='PackClsInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='mmcls.ToPIL', to_rgb=True),
    dict(type='mmcls.torchvision/Resize', size=256, interpolation=3),
    dict(type='mmcls.torchvision/CenterCrop', size=224),
    dict(type='mmcls.ToNumpy', to_rgb=True),
    dict(type='PackClsInputs'),
]
train_dataloader = dict(
    batch_size=2048, dataset=dict(pipeline=train_pipeline), drop_last=True)
val_dataloader = dict(dataset=dict(pipeline=test_pipeline), drop_last=False)
test_dataloader = dict(dataset=dict(pipeline=test_pipeline), drop_last=False)