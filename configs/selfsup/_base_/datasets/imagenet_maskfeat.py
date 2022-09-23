# dataset settings
data_source = 'ImageNet'
dataset_type = 'SingleViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(
        type='RandomResizedCropAndInterpolationWithTwoPic',
        size=224,
        scale=(0.5, 1.0),
        ratio=(0.75, 1.3333),
        interpolation='bicubic'),
    dict(type='RandomHorizontalFlip')
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

train_pipeline.append(dict(type='MaskFeatMaskGenerator', mask_ratio=0.4))

# dataset summary
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train',
            ann_file='data/imagenet/meta/train.txt'),
        pipeline=train_pipeline,
        prefetch=prefetch))
