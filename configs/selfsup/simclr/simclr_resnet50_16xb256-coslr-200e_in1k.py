_base_ = 'simclr_resnet50_8xb32-coslr-200e_in1k.py'

# optimizer
optimizer = dict(lr=4.8)
optim_wrapper = dict(optimizer=optimizer)

# dataset summary
train_dataloader = dict(batch_size=256)
