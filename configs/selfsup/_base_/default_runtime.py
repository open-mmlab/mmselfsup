default_scope = 'mmselfsup'

default_hooks = dict(
    optimizer=dict(type='OptimizerHook', grad_clip=None),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(
    interval=50,
    custom_keys=[dict(data_src='', method='mean', windows_size='global')])

# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='SelfSupLocalVisualizer',
#     vis_backends=vis_backends,
#     name='visualizer')

# custom_hooks = [dict(type='SelfSupVisualizationHook', interval=10)]

log_level = 'INFO'
load_from = None
resume = False
