# dataset settings
custom_imports = dict(imports='mmcls.datasets', allow_failed_imports=False)
dataset_type = 'mmcls.ImageNet'
data_root = 'data/imagenet/'
# file_client_args = dict(backend='disk')
# file_client_args = dict(
#     backend='petrel',
#     # 因为petreloss.conf设置sproject为默认，此处可省略前缀
#     path_mapping=dict({
#         './data/imagenet': 's3://openmmlab/datasets/classification/imagenet',
#         'data/imagenet': 's3://openmmlab/datasets/classification/imagenet'
#     })
# )
file_client_args = dict(
    backend='memcached',
    server_list_cfg='/mnt/lustre/share/memcached_client/pcs_server_list.conf',
    client_cfg='/mnt/lustre/share_data/zhangwenwei/software/pymc/mc.conf',
    sys_path='/mnt/lustre/share_data/zhangwenwei/software/pymc')

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RandomResizedCrop',
        size=224,
        scale=(0.2, 1.0),
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSelfSupInputs', meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=128,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='meta/train.txt',
        data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))
