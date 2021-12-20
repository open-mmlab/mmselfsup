config = 'configs/selfsup/mae/mae_vit-base_b4096-coslr-100e_in1k.py'
work_dirs = 'work_dirs/selfsup/mae_imagenet_pretrain_b4096_100e_16gpus'
job_name = 'mae_4096_100'
srun_args = '--quotatype=spot'
partition = 'mm_dev'
gpus_per_node = 8
gpus = 16
