# Copyright (c) OpenMMLab. All rights reserved.
import mmcls
import mmcv
from mmengine.utils import digit_version

from .version import __version__

mmcv_minimum_version = '2.0.0rc1'
mmcv_maximum_version = '2.1.0'
mmcv_version = digit_version(mmcv.__version__)

mmcls_minimum_version = '1.0.0rc0'
mmcls_maximum_version = '1.1.0'
mmcls_version = digit_version(mmcls.__version__)


assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version < digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}.'

assert (mmcls_version >= digit_version(mmcls_minimum_version)
        and mmcls_version < digit_version(mmcls_maximum_version)), \
    f'MMClassification=={mmcls.__version__} is used but incompatible. ' \
    f'Please install mmcls>={mmcls_minimum_version}, <{mmcls_maximum_version}.'

__all__ = ['__version__']
