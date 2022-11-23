_base_ = [
    '../../../configs/selfsup/mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py'  # noqa:E501
]

custom_imports = dict(imports=['projects.example_project'])

_base_.model.type = 'DummyMAE'
