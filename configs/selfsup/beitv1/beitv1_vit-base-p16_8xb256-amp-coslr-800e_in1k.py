_base_ = 'beit_vit-base-p16_32xb64-amp-coslr-800e_in1k.py'

# dataset 8GPUs x 256
train_dataloader = dict(batch_size=256, num_workers=16)
