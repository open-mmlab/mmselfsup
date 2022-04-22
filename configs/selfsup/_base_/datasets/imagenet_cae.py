# dataset settings
data_source = 'ImageNet'
dataset_type = 'SingleViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomHorizontalFlip', p=0.5),
    dict(
        type='RandomResizedCropAndInterpolationWithTwoPic',
        size=224,
        second_size=112,
        interpolation='bicubic',
        second_interpolation='lanczos',
        scale=(0.08, 1.0)),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend([dict(type='ToTensor')])

train_pipeline.append(
    dict(
        type='BEiTMaskGenerator',
        input_size=(14, 14),
        num_masking_patches=75,
        max_num_patches=None,
        min_num_patches=16))

# dataset summary
file_client_args = dict(
    backend='memcached',
    server_list_cfg='/mnt/lustre/share/memcached_client/pcs_server_list.conf',
    client_cfg='/mnt/lustre/share_data/zhangwenwei/software/pymc/mc.conf',
    sys_path='/mnt/lustre/share_data/zhangwenwei/software/pymc')
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/imagenet/train',
            ann_file='data/imagenet/meta/train.txt',
            file_client_args=file_client_args),
        pipeline=train_pipeline,
        prefetch=prefetch))
