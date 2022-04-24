import argparse
import json
import os
import os.path as osp
import re
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

from modelindex.load_model_index import load
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

console = Console()
METRICS_MAP = {'Top 1 Accuracy': 'top1', 'Top 5 Accuracy': 'top5'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train models (in bench_train.yml) and compare accuracy.')
    parser.add_argument(
        'partition', type=str, help='Cluster partition to use.')
    parser.add_argument(
        '--job-name',
        type=str,
        default='selfsup-pretrain-benchmark',
        help='Slurm job name prefix')
    parser.add_argument('--port', type=int, default=29777, help='dist port')
    parser.add_argument(
        '--models', nargs='+', type=str, help='Specify model names to run.')
    parser.add_argument(
        '--work-dir',
        default='work_dirs/benchmark_pretrain_with_task',
        help='the dir to save train log')
    parser.add_argument(
        '--run', action='store_true', help='run script directly')
    parser.add_argument(
        '--local',
        action='store_true',
        help='run at local instead of cluster.')
    parser.add_argument(
        '--mail', type=str, help='Mail address to watch train status.')
    parser.add_argument(
        '--mail-type',
        nargs='+',
        default=['BEGIN', 'END', 'FAIL'],
        choices=['NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL'],
        help='Mail address to watch train status.')
    parser.add_argument(
        '--quotatype',
        default=None,
        choices=['reserved', 'auto', 'spot'],
        help='Quota type, only available for phoenix-slurm>=0.2')
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Summarize benchmark train results.')
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save the summary and archive log files.')

    args = parser.parse_args()
    return args


def create_train_job_batch(commands, model_info, args, port, script_name):

    fname = model_info.name
    parent_work_dir = Path(args.work_dir) / fname

    # common settings
    if args.mail is not None and 'NONE' not in args.mail_type:
        mail_cfg = (f'#SBATCH --mail {args.mail}\n'
                    f'#SBATCH --mail-type {args.mail_type}\n')
    else:
        mail_cfg = ''

    if args.quotatype is not None:
        quota_cfg = f'#SBATCH --quotatype {args.quotatype}\n'
    else:
        quota_cfg = ''

    launcher = 'none' if args.local else 'slurm'
    runner = 'python' if args.local else 'srun python'

    # pretrain task
    assert 'PretrainGPUs' in model_info.data, \
        f"Haven't specify gpu numbers for {fname}"
    gpus = model_info.data['PretrainGPUs']

    config_path = model_info.data['PretrainConfig']
    config = Path(config_path)
    assert config.exists(), f'"{fname}": {config} not found.'

    job_name = f'{args.job_name}_{fname}'
    pretrain_work_dir = parent_work_dir / Path(
        config_path.split('/')[-1].replace('.py', ''))
    pretrain_work_dir.mkdir(parents=True, exist_ok=True)

    job_script = (f'#!/bin/bash\n'
                  f'#SBATCH --output {pretrain_work_dir}/job.%j.out\n'
                  f'#SBATCH --partition={args.partition}\n'
                  f'#SBATCH --job-name {job_name}\n'
                  f'#SBATCH --gres=gpu:8\n'
                  f'{mail_cfg}{quota_cfg}'
                  f'#SBATCH --ntasks-per-node=8\n'
                  f'#SBATCH --ntasks={gpus}\n'
                  f'#SBATCH --cpus-per-task=12\n\n'
                  f'{runner} -u {script_name} {config} '
                  f'--work-dir={pretrain_work_dir} --cfg-option '
                  f'dist_params.port={port} '
                  f'checkpoint_config.max_keep_ckpts=1 '
                  f'--launcher={launcher}\n')

    # extract backbone weights
    extract_script = (
        f'\n'
        f'python tools/model_converters/extract_backbone_weights.py '
        f'{pretrain_work_dir}/latest.pth '
        f'{parent_work_dir}/extracted_backbone.pth\n')
    weights = parent_work_dir / 'extracted_backbone.pth'

    # downstream task
    linear_config = model_info.data.get('LinearConfig', None)
    if linear_config:
        assert 'LinearGPUs' in model_info.data, \
            f"Haven't specify gpu numbers for {fname}"
        linear_gpus = model_info.data['LinearGPUs']

        linear_config_file = Path(linear_config)
        assert linear_config_file.exists(), \
            f'"{fname}": {linear_config_file} not found.'

        job_name = f'{args.job_name}_{fname}'
        linear_work_dir = parent_work_dir / Path(
            linear_config.split('/')[-1].replace('.py', ''))
        linear_work_dir.mkdir(parents=True, exist_ok=True)

        linear_job_script = (f'\n'
                             f'#SBATCH --output {linear_work_dir}/job.%j.out\n'
                             f'#SBATCH --partition={args.partition}\n'
                             f'#SBATCH --job-name {job_name}\n'
                             f'#SBATCH --gres=gpu:8\n'
                             f'{mail_cfg}{quota_cfg}'
                             f'#SBATCH --ntasks-per-node=8\n'
                             f'#SBATCH --ntasks={linear_gpus}\n'
                             f'#SBATCH --cpus-per-task=12\n\n'
                             f'{runner} -u {script_name} {linear_config_file} '
                             f'--work-dir={linear_work_dir} --cfg-option '
                             f'dist_params.port={port} '
                             f'model.backbone.init_cfg.type=Pretrained '
                             f'model.backbone.init_cfg.checkpoint={weights} '
                             f'checkpoint_config.max_keep_ckpts=1 '
                             f'--launcher={launcher}\n')

    fine_tuning_config = model_info.data.get('FinetuningConfig', None)
    if fine_tuning_config:
        assert 'FinetuningGPUs' in model_info.data, \
            f"Haven't specify gpu numbers for {fname}"
        fine_tuning_gpus = model_info.data['FinetuningGPUs']

        fine_tuning_config_file = Path(fine_tuning_config)
        assert fine_tuning_config_file.exists(), \
            f'"{fname}": {fine_tuning_config_file} not found.'

        job_name = f'{args.job_name}_{fname}'
        fine_tuning_work_dir = parent_work_dir / Path(
            fine_tuning_config.split('/')[-1].replace('.py', ''))
        fine_tuning_work_dir.mkdir(parents=True, exist_ok=True)

        fine_tuning_job_script = (
            f'\n'
            f'#SBATCH --output {fine_tuning_work_dir}/job.%j.out\n'
            f'#SBATCH --partition={args.partition}\n'
            f'#SBATCH --job-name {job_name}\n'
            f'#SBATCH --gres=gpu:8\n'
            f'{mail_cfg}{quota_cfg}'
            f'#SBATCH --ntasks-per-node=8\n'
            f'#SBATCH --ntasks={fine_tuning_gpus}\n'
            f'#SBATCH --cpus-per-task=12\n\n'
            f'{runner} -u {script_name} {fine_tuning_config_file} '
            f'--work-dir={fine_tuning_work_dir} --cfg-option '
            f'dist_params.port={port} '
            f'model.backbone.init_cfg.type=Pretrained '
            f'model.backbone.init_cfg.checkpoint={weights} '
            f'checkpoint_config.max_keep_ckpts=1 '
            f'--launcher={launcher}\n')

    with open(parent_work_dir / 'job.sh', 'w') as f:
        f.write(job_script)
        f.write(extract_script)
        if linear_config:
            f.write(linear_job_script)

        if fine_tuning_config:
            f.write(fine_tuning_job_script)

    commands.append(f'echo "{config}"')
    if args.local:
        commands.append(f'bash {parent_work_dir}/job.sh')
    else:
        commands.append(f'sbatch {parent_work_dir}/job.sh')

    return parent_work_dir / 'job.sh'


def train(args):
    models_cfg = load(str(Path(__file__).parent / 'benchmark_models.yml'))
    models_cfg.build_models_with_collections()
    models = {model.name: model for model in models_cfg.models}

    script_name = osp.join('tools', 'train.py')
    port = args.port

    commands = []
    if args.models:
        patterns = [re.compile(pattern) for pattern in args.models]
        filter_models = {}
        for k, v in models.items():
            if any([re.match(pattern, k) for pattern in patterns]):
                filter_models[k] = v
        if len(filter_models) == 0:
            print('No model found, please specify models in:')
            print('\n'.join(models.keys()))
            return
        models = filter_models

    for model_info in models.values():
        months = model_info.data.get('Months', range(1, 13))
        if datetime.now().month in months:
            _ = create_train_job_batch(commands, model_info, args, port,
                                       script_name)
            port += 1

    print(commands)

    command_str = '\n'.join(commands)

    preview = Table()
    preview.add_column('Shell command preview')
    preview.add_row(
        Syntax(
            command_str,
            'bash',
            background_color='default',
            line_numbers=True,
            word_wrap=True))
    console.print(preview)

    if args.run:
        os.system(command_str)
    else:
        console.print('Please set "--run" to start the job')


def save_summary(summary_data, models_map, work_dir):
    date = datetime.now().strftime('%Y%m%d-%H%M%S')
    zip_path = work_dir / f'archive-{date}.zip'
    zip_file = ZipFile(zip_path, 'w')
    summary_path = work_dir / 'benchmark_summary.md'
    file = open(summary_path, 'w')
    headers = [
        'Model', 'Top-1 Expected(%)', 'Top-1 (%)', 'Top-1 best(%)',
        'best epoch', 'Top-5 Expected (%)', 'Top-5 (%)', 'Config', 'Log'
    ]
    file.write('# Train Benchmark Regression Summary\n')
    file.write('| ' + ' | '.join(headers) + ' |\n')
    file.write('|:' + ':|:'.join(['---'] * len(headers)) + ':|\n')
    for model_name, summary in summary_data.items():
        if len(summary) == 0:
            # Skip models without results
            continue
        row = [model_name]
        if 'Top 1 Accuracy' in summary:
            metric = summary['Top 1 Accuracy']
            row.append(f"{metric['expect']:.2f}")
            row.append(f"{metric['last']:.2f}")
            row.append(f"{metric['best']:.2f}")
            row.append(f"{metric['best_epoch']:.2f}")
        else:
            row.extend([''] * 4)
        if 'Top 5 Accuracy' in summary:
            metric = summary['Top 5 Accuracy']
            row.append(f"{metric['expect']:.2f}")
            row.append(f"{metric['last']:.2f}")
        else:
            row.extend([''] * 2)

        model_info = models_map[model_name]
        row.append(model_info.config)
        row.append(str(summary['log_file'].relative_to(work_dir)))
        zip_file.write(summary['log_file'])
        file.write('| ' + ' | '.join(row) + ' |\n')
    file.close()
    zip_file.write(summary_path)
    zip_file.close()
    print('Summary file saved at ' + str(summary_path))
    print('Log files archived at ' + str(zip_path))


def show_summary(summary_data):
    table = Table(title='Train Benchmark Regression Summary')
    table.add_column('Model')
    for metric in METRICS_MAP:
        table.add_column(f'{metric} (expect)')
        table.add_column(f'{metric}')
        table.add_column(f'{metric} (best)')

    def set_color(value, expect):
        if value > expect:
            return 'green'
        elif value > expect - 0.2:
            return 'white'
        else:
            return 'red'

    for model_name, summary in summary_data.items():
        row = [model_name]
        for metric_key in METRICS_MAP:
            if metric_key in summary:
                metric = summary[metric_key]
                expect = metric['expect']
                last = metric['last']
                last_color = set_color(last, expect)
                best = metric['best']
                best_color = set_color(best, expect)
                best_epoch = metric['best_epoch']
                row.append(f'{expect:.2f}')
                row.append(f'[{last_color}]{last:.2f}[/{last_color}]')
                row.append(
                    f'[{best_color}]{best:.2f}[/{best_color}] ({best_epoch})')
        table.add_row(*row)

    console.print(table)


def summary(args):
    models_cfg = load(str(Path(__file__).parent / 'benchmark_models.yml'))
    models = {model.name: model for model in models_cfg.models}

    work_dir = Path(args.work_dir)
    dir_map = {p.name: p for p in work_dir.iterdir() if p.is_dir()}

    if args.models:
        patterns = [re.compile(pattern) for pattern in args.models]
        filter_models = {}
        for k, v in models.items():
            if any([re.match(pattern, k) for pattern in patterns]):
                filter_models[k] = v
        if len(filter_models) == 0:
            print('No model found, please specify models in:')
            print('\n'.join(models.keys()))
            return
        models = filter_models

    summary_data = {}
    for model_name, model_info in models.items():

        # Skip if not found any log file.
        if model_name not in dir_map:
            summary_data[model_name] = {}
            continue
        sub_dir = dir_map[model_name]
        log_files = list(sub_dir.glob('*.log.json'))
        if len(log_files) == 0:
            continue
        log_file = sorted(log_files)[-1]

        # parse train log
        with open(log_file) as f:
            json_logs = [json.loads(s) for s in f.readlines()]
            val_mode_logs = [
                log for log in json_logs
                if 'mode' in log and log['mode'] == 'val'
            ]

        if len(val_mode_logs) == 0:
            continue

        expect_metrics = model_info.results[0].metrics

        # extract metrics
        summary = {'log_file': log_file}
        for key_yml, key_res in METRICS_MAP.items():
            if key_yml in expect_metrics:
                for key_log, _ in val_mode_logs[-1].items():
                    if key_res in key_log:
                        assert key_log in val_mode_logs[-1], \
                            f'{model_name}: No metric "{key_log}"'
                        expect_result = float(expect_metrics[key_yml])
                        last = float(val_mode_logs[-1][key_log])
                        best_log = sorted(
                            val_mode_logs, key=lambda x: x[key_log])[-1]
                        best = float(best_log[key_log])
                        best_epoch = int(best_log['epoch'])

                summary[key_yml] = dict(
                    expect=expect_result,
                    last=last,
                    best=best,
                    best_epoch=best_epoch)
        summary_data[model_name] = summary

    show_summary(summary_data)
    if args.save:
        save_summary(summary_data, models, work_dir)


def main():
    args = parse_args()

    if args.summary:
        summary(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
