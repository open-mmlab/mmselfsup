from mae_b4096_16gpus_100e import *

nohup = True
output_file = 'test_2.txt'
resume_from = None

if nohup:
    if resume_from is not None:
        cmd = f"GPUS_PER_NODE={gpus_per_node} GPUS={gpus} SRUN_ARGS={f'{srun_args}'} \
            nohup bash tools/slurm_train.sh {partition} \
            {job_name} {config} {work_dirs} --resume_from {resume_from} \
            > f{output_file} &"

    else:
        cmd = f"GPUS_PER_NODE={gpus_per_node} GPUS={gpus} SRUN_ARGS={f'{srun_args}'} \
            nohup bash tools/slurm_train.sh {partition} \
            {job_name} {config} {work_dirs} > f{output_file} &"

else:
    if resume_from is not None:
        cmd = f"GPUS_PER_NODE={gpus_per_node} GPUS={gpus} SRUN_ARGS={f'{srun_args}'} \
            bash tools/slurm_train.sh {partition} \
            {job_name} {config} {work_dirs} --resume_from {resume_from}"

    else:
        cmd = f"GPUS_PER_NODE={gpus_per_node} GPUS={gpus} SRUN_ARGS={f'{srun_args}'} \
            bash tools/slurm_train.sh {partition} \
            {job_name} {config} {work_dirs}"

print(cmd)
