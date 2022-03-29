_base_ = 'swin-base_ft-8xb256-coslr-100e_in1k.py'

data = dict(samples_per_gpu=64, workers_per_gpu=8)
