# dataset settings
data_source = 'ImageNet'
dataset_type = 'SingleViewWithHogDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

pipeline_before_hog = [
    dict(
        type='RandomResizedCrop', size=224, scale=(0.5, 1.0), interpolation=3),
    dict(type='RandomHorizontalFlip')
]
pipeline_after_hog = [
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]

hog_para = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(1, 1),
    block_norm='L2',
    patch_size=16)

# prefetch
prefetch = False

# dataset summary
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train',
            ann_file='data/imagenet/meta/train.txt',
        ),
        pipeline_before_hog=pipeline_before_hog,
        pipeline_after_hog=pipeline_after_hog,
        hog_para=hog_para,
        prefetch=prefetch))
