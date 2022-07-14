_base_ = 'deepcluster_resnet50_8xb64-steplr-200e_in1k.py'

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNetSobel',
        depth=50,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')))
