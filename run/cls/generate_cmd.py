from mae_finetune_100e import *

cmd = f'GPUS_PER_NODE={gpus_per_node} GPUS={gpus} SRUN_ARGS="{srun_args}" \
    bash tools/benchmarks/classification/slurm_train_linear.sh {partition} \
    {job_name} {config} {pretrain}'

print(cmd)
