_base_ = 'vit-base-p16_ft-8xb128-coslr-100e_in1k.py'

# model settings
# MAE ViT-large set drop_path_rate to 0.2
model = dict(
    backbone=dict(arch='large', drop_path_rate=0.2),
    head=dict(in_channels=1024))

# optim settings
# learning rate and layer decay rate are set to 0.004 and 0.75 respectively
optim_wrapper = dict(optimizer=dict(lr=0.004, layer_decay_rate=0.75))

# training cfg
# fine-tuning for 50 epochs for ViT-large
train_cfg = dict(max_epochs=50)
