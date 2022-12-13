_base_ = [
    'mmselfsup::selfsup/mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py'
]

custom_imports = dict(imports=['models'])

_base_.model.type = 'DummyMAE'
