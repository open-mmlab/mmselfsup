data_source_cfg = dict(type='ImageList', memcached=False, mclient_path=None)
data_root = "data/VOCdevkit/VOC2007/JPEGImages"
data_all_list = "data/VOCdevkit/VOC2007/Lists/trainvaltest.txt"
split_at = [5011]
split_name = ['voc07_trainval', 'voc07_test']
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data = dict(
    imgs_per_gpu=32,
    workers_per_gpu=2,
    extract=dict(
        type="ExtractDataset",
        data_source=dict(
            list_file=data_all_list, root=data_root, **data_source_cfg),
        pipeline=[
            dict(type='Resize', size=256),
            dict(type='Resize', size=(224, 224)),
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
        ]))
