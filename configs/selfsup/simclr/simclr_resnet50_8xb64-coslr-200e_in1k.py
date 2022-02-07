_base_ = 'simclr_resnet50_8xb32-coslr-200e_in1k.py'

# dataset summary
data = dict(samples_per_gpu=64)  # total 64*8
