_base_ = 'vit-base-p16_linear-8xb2048-coslr-90e_in1k.py'

# optimizer
optimizer = dict(type='SGD', lr=1.6, weight_decay=0.0, momentum=0.9)
optim_wrapper = dict(optimizer=optimizer)
