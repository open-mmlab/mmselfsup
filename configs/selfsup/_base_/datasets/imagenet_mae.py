# dataset settings
data_source = 'ImageNet'
dataset_type = 'MAEDataset'
file_client_args = dict()
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
target_img_norm_cfg = dict(mean=[-2.1179, -2.0357, -1.8044], std=[4.3668, 4.4643, 4.4444])
train_pipeline = [dict(type='RandomResizedCrop', size=224)]
train_target_pipeline = []

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg)])
    train_target_pipeline.extend([dict(type='Normalize', **target_img_norm_cfg)])
    

# dataset summary
data = dict(
    imgs_per_gpu=128,  # total 32*8
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='/mnt/lustre/share_data/openmmlab/datasets/classification/imagenet/train',
            ann_file='/mnt/lustre/share_data/openmmlab/datasets/classification/imagenet/meta/train.txt',
        ),
        pipeline=train_pipeline,
        target_pipeline=train_target_pipeline,
        prefetch=prefetch,
        mask_ratio=0.75,
        window_size=224 // 16
    ))
