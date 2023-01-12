import copy

# dataset settings
dataset_type = 'ORLDataset'
meta_json = '../data/coco/meta/train2017_10nn_instance_correspondence.json'
data_train_root = '../data/coco/train2017'
# file_client_args = dict(backend='disk')
view_pipeline = [
    dict(
        type='RandomResizedCrop',
        size=224,
        interpolation='bicubic',
        backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989),
        color_format='rgb'),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=1.),
    dict(type='RandomSolarize', prob=0)
]

view_patch_pipeline = [
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        prob=0.8),
    dict(
        type='RandomGrayscale',
        prob=0.2,
        keep_channels=True,
        channel_weights=(0.114, 0.587, 0.2989),
        color_format='rgb'),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=1.),
    dict(type='RandomSolarize', prob=0)
]
view_pipeline1 = copy.deepcopy(view_pipeline)
view_pipeline2 = copy.deepcopy(view_pipeline)
view_patch_pipeline1 = copy.deepcopy(view_patch_pipeline)
view_patch_pipeline2 = copy.deepcopy(view_patch_pipeline)
view_pipeline2[4]['prob'] = 0.1  # gaussian blur
view_pipeline2[5]['prob'] = 0.2  # solarization
view_patch_pipeline1[3]['prob'] = 0.1  # gaussian blur
view_patch_pipeline2[4]['prob'] = 0.2  # solarization

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        root=data_train_root,
        json_file=meta_json,
        topk_knn_image=10,
        img_pipeline1=view_pipeline1,
        img_pipeline2=view_pipeline2,
        patch_pipeline1=view_patch_pipeline1,
        patch_pipeline2=view_patch_pipeline2,
        patch_size=96,
        interpolation=2,
        shift=(-0.5, 0.5),
        scale=(0.5, 2.),
        ratio=(0.5, 2.),
        iou_thr=0.5,
        attempt_num=200,
    ))
