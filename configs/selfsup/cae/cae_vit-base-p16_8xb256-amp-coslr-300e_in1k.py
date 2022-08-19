_base_ = 'cae_vit-base-p16_16xb128-amp-coslr-300e_in1k.py'

# dataset 8GPUs x 256
train_dataloader = dict(batch_size=256, num_workers=16)
