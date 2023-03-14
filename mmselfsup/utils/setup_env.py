# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """Register all modules in mmselfsup into the registries.

    Args:
        init_default_scope (bool): Whether initialize the mmselfsup default
            scope. When `init_default_scope=True`, the global default scope
            will be set to `mmselfsup`, and all registries will build modules
            from mmselfsup's registry node. To understand more about the
            registry, please refer to
            https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/registry.md
            Defaults to True.
    """  # noqa
    import mmselfsup.datasets  # noqa: F401,F403
    import mmselfsup.engine  # noqa: F401,F403
    import mmselfsup.evaluation  # noqa: F401,F403
    import mmselfsup.models  # noqa: F401,F403
    import mmselfsup.structures  # noqa: F401,F403
    import mmselfsup.visualization  # noqa: F401,F403

    if init_default_scope:
        never_created = DefaultScope.get_current_instance() is None \
                        or not DefaultScope.check_instance_created('mmselfsup')
        if never_created:
            DefaultScope.get_instance('mmselfsup', scope_name='mmselfsup')
            return
        current_scope = DefaultScope.get_current_instance()
        if current_scope.scope_name != 'mmselfsup':
            warnings.warn('The current default scope '
                          f'"{current_scope.scope_name}" is not "mmselfsup", '
                          '`register_all_modules` will force set the current'
                          'default scope to "mmselfsup". If this is not as '
                          'expected, please set `init_default_scope=False`.')
            # avoid name conflict
            new_instance_name = f'mmselfsup-{datetime.datetime.now()}'
            DefaultScope.get_instance(
                new_instance_name, scope_name='mmselfsup')
