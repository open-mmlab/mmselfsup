config='configs/selfsup/mae/mae_vit-base_b128-coslr-400e_in1k.py'
work_dirs='work_dirs/selfsup/mae_imagenet_pretrain_b128_32gpus'
job_name='mae_128_32'
srun_args='--quotatype=spot'
partition='mm_dev'
gpus_per_node=8
gpus=32
