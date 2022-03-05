_base_ = 'simmim_swin-base_16xb128-coslr-100e_in1k.py'

# data
data = dict(samples_per_gpu=256)
