_base_ = 'simclr_resnet50_8xb32-coslr-200e_in1k.py'

# optimizer
optimizer = dict(lr=4.8)

# dataset summary
data = dict(samples_per_gpu=256, workers_per_gpu=8)  # total 256*16
