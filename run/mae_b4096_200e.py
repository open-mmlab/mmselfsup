config = 'configs/selfsup/mae/mae_vit-base_b4096-coslr-200e_in1k.py'
work_dirs = 'work_dirs/selfsup/mae_imagenet_pretrain_b4096_200e'
job_name = 'mae_4096_200'
srun_args = '--quotatype=spot --async'
partition = 'mm_model'
gpus_per_node = 8
gpus = 32
