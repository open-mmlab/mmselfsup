_base_ = 'mae_vit-base-p16_8xb512-coslr-400e_in1k.py'

# mixed precision
fp16 = dict(loss_scale='dynamic')
