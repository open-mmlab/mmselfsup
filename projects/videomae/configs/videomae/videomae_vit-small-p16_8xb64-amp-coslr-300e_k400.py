_base_ = [
    '../_base_/models/videomae_vit-small-p16.py',
    '../_base_/datasets/k400_videomae.py',
    '../_base_/schedules/adamw_coslr-100e_in1k.py',
    'mmselfsup::_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['mmaction2.datasets.transforms', 'models', 'datasets'],
    allow_failed_imports=False)
