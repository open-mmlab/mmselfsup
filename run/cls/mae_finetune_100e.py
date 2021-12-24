config = 'configs/benchmarks/classification/imagenet/vit-b-16_16xb64-coslr-100e_in1k.py'
job_name = 'mae_ft_16xb64'
srun_args = '--quotatype=spot --async'
partition = 'mm_model'
pretrain = '/mnt/lustre/liuyuan1.vendor/ckpt/mae/mae_imagenet_pretrain_b4096_400e/epoch_400.pth'
gpus_per_node = 8
gpus = 16
