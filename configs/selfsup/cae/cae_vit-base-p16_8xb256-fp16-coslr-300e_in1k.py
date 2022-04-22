_base_ = 'cae_vit-base-p16_16xb128-fp16-coslr-300e_in1k.py'

# dataset
data = dict(samples_per_gpu=256, workers_per_gpu=8)
