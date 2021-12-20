from mae_b128_32gpus import *

nohup=True
output_file='test_6.txt'
resume_from="/mnt/lustre/liuyuan1.vendor/ckptp/mae/mae_imagenet_pretrain_b128_32gpus/epoch_12.pth"

if nohup:
    cmd = f"GPUS_PER_NODE={gpus_per_node} GPUS={gpus} SRUN_ARGS={f'{srun_args}'} \
        nohup bash tools/slurm_train.sh {partition} \
        {job_name} {config} {work_dirs} --resume_from {resume_from} \
        > f{output_file} &"

else:
    cmd = f"GPUS_PER_NODE={gpus_per_node} GPUS={gpus} SRUN_ARGS={f'{srun_args}'} \
        bash tools/slurm_train.sh {partition} \
        {job_name} {config} {work_dirs} --resume_from {resume_from}"


print(cmd)
