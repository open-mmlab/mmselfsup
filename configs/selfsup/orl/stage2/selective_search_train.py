_base_ = [
    '../../_base_/default_runtime.py',
]

# model settings
# done
model = dict(type='SelectiveSearch')
dist_params = dict(backend='nccl', port=29500)
# dataset settings
data_train_json = '../data/coco/annotations/instances_train2017.json'
data_train_root = '../data/coco/train2017'
dataset_type = 'SSDataset'

val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        root=data_train_root,
        json_file=data_train_json,
        method='fast',
        min_size=None,
        max_ratio=None,
        topN=None))
