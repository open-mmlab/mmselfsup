# Copyright (c) OpenMMLab. All rights reserved.
import mmcls
import mmcv
import mmengine
from mmengine.utils import digit_version

from .version import __version__

mmcv_minimum_version = '1.4.2'
mmcv_maximum_version = '1.7.0'
mmcv_version = digit_version(mmcv.__version__)

mmengine_minimum_version = '0.0.0'
mmengine_maximum_version = '0.2.0'
mmengine_version = digit_version(mmengine.__version__)

mmcls_minimum_version = '0.21.0'
mmcls_version = digit_version(mmcls.__version__)


assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version <= digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <={mmcv_maximum_version}.'

assert (mmengine_version >= digit_version(mmengine_minimum_version)
        and mmengine_version <= digit_version(mmengine_maximum_version)), \
    f'MMEngine=={mmengine.__version__} is used but incompatible. ' \
    f'Please install mmengine>={mmengine_minimum_version}, ' \
    f'<={mmengine_maximum_version}.'

assert mmcls_version >= digit_version(mmcls_minimum_version), \
    f'MMClassification=={mmcls.__version__} is used but incompatible. ' \
    f'Please install mmcls>={mmcls_minimum_version}.'

__all__ = ['__version__']
