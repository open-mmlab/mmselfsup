_base_ = '../mae/mae_vit-base-p16_8xb512-amp-coslr-400e_in1k.py'

# model settings
model = dict(
    type='MILAN',
    backbone=dict(
        type='MILANViT',
        init_cfg=[
            dict(type='Xavier', distribution='uniform', layer='Linear'),
            dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
        ]),
    neck=dict(
        type='MILANPretrainDecoder',
        init_cfg=[
            dict(type='Xavier', distribution='uniform', layer='Linear'),
            dict(type='Constant', layer='LayerNorm', val=1.0, bias=0.0)
        ]),
    head=dict(
        _delete_=True,
        type='MILANPretrainHead',
        loss=dict(type='MILANReconstructionLoss')),
    target_generator=dict(
        type='CLIPGenerator',
        tokenizer_path=  # noqa
        './milan_ckpt/clip_vit_base_16.pth.tar'),
    init_cfg=None)

# dataset 8 x 512
train_dataloader = dict(batch_size=256, num_workers=16)

find_unused_parameters = True
