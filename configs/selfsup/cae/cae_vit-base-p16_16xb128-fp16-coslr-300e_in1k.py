_base_ = 'cae_vit-base-p16_32xb64-fp16-coslr-300e_in1k.py'

# dataset
data = dict(samples_per_gpu=128, workers_per_gpu=8)
