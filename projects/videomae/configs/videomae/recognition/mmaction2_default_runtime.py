# default_scope = 'mmaction'

default_hooks = dict(
    runtime_info=dict(type='mmaction.RuntimeInfoHook'),
    timer=dict(type='mmaction.IterTimerHook'),
    logger=dict(type='mmaction.LoggerHook', interval=20, ignore_last=False),
    param_scheduler=dict(type='mmaction.ParamSchedulerHook'),
    checkpoint=dict(
        type='mmaction.CheckpointHook', interval=1, save_best='auto'),
    sampler_seed=dict(type='mmaction.DistSamplerSeedHook'),
    sync_buffers=dict(type='mmaction.SyncBuffersHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

log_processor = dict(
    type='mmaction.LogProcessor', window_size=20, by_epoch=True)

vis_backends = [dict(type='mmaction.LocalVisBackend')]
visualizer = dict(type='mmaction.ActionVisualizer', vis_backends=vis_backends)

log_level = 'INFO'
load_from = None
resume = False
