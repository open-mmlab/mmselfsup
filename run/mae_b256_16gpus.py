config='configs/selfsup/mae/mae_vit-base_b256-coslr-400e_in1k.py'
work_dirs='work_dirs/selfsup/mae_imagenet_pretrain_b256_16gpus'
job_name='mae_256_16'
srun_args='--quotatype=spot'
partition='mm_dev'
gpus_per_node=8
gpus=16
