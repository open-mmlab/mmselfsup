_base_ = 'mae_vit-b-16_8xb512-coslr-400e_in1k.py'

# schedule
runner = dict(max_epochs=1600)
