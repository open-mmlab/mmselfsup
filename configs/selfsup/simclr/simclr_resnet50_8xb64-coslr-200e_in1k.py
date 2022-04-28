_base_ = 'simclr_resnet50_8xb32-coslr-200e_in1k.py'

# optimizer
optimizer = dict(lr=0.6)

# dataset summary
data = dict(samples_per_gpu=64)  # total 64*8
