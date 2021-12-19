from mae_b256_16gpus import *

cmd = f"GPUS_PER_NODE={gpus_per_node} GPUS={gpus} SRUN_ARGS={f'{srun_args}'} \
    bash tools/slurm_train.sh {partition} \
    {job_name} {config} {work_dirs}"

print(cmd)