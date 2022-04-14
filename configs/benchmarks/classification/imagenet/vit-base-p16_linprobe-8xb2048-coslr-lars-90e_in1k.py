_base_ = 'vit-base-p16_linprobe-16xb1024-coslr-lars-90e_in1k.py'

# dataset
data = dict(imgs_per_gpu=2048, workers_per_gpu=16)
