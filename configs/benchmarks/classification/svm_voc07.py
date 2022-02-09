data_source = 'ImageList'
dataset_type = 'SingleViewDataset'
split_at = [5011]
split_name = ['voc07_trainval', 'voc07_test']
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    extract=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='data/VOCdevkit/VOC2007/JPEGImages',
            ann_file='data/VOCdevkit/VOC2007/Lists/trainvaltest.txt',
        ),
        pipeline=[
            dict(type='Resize', size=256),
            dict(type='Resize', size=(224, 224)),
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
        ]))
