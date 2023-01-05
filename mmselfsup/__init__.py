# Copyright (c) OpenMMLab. All rights reserved.
import mmcls
import mmcv
import mmengine
from mmengine.utils import digit_version

from .version import __version__

mmengine_minimum_version = '0.3.0'
mmengine_maximum_version = '1.0.0'
mmengine_version = digit_version(mmengine.__version__)

<<<<<<< HEAD
mmcv_minimum_version = '2.0.0rc1'
mmcv_maximum_version = '2.1.0'
mmcv_version = digit_version(mmcv.__version__)

mmcls_minimum_version = '1.0.0rc4'
mmcls_maximum_version = '1.1.0'
=======
def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.

    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.

    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Defaults to 4.

    Returns:
        tuple[int]: The version info in digits (integers).
    """
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])
    else:
        release.extend([0, 0])
    return tuple(release)


mmcv_minimum_version = '1.4.2'
mmcv_maximum_version = '1.9.0'
mmcv_version = digit_version(mmcv.__version__)

mmcls_minimum_version = '0.21.0'
mmcls_maximum_version = '0.27.0'
>>>>>>> upstream/master
mmcls_version = digit_version(mmcls.__version__)

assert (mmengine_version >= digit_version(mmengine_minimum_version)
        and mmengine_version < digit_version(mmengine_maximum_version)), \
    f'MMEngine=={mmengine.__version__} is used but incompatible. ' \
    f'Please install mmengine>={mmengine_minimum_version}, ' \
    f'<{mmengine_maximum_version}.'

assert (mmcv_version >= digit_version(mmcv_minimum_version)
        and mmcv_version < digit_version(mmcv_maximum_version)), \
    f'MMCV=={mmcv.__version__} is used but incompatible. ' \
    f'Please install mmcv>={mmcv_minimum_version}, <{mmcv_maximum_version}.'

assert (mmcls_version >= digit_version(mmcls_minimum_version)
        and mmcls_version < digit_version(mmcls_maximum_version)), \
    f'MMClassification=={mmcls.__version__} is used but incompatible. ' \
    f'Please install mmcls>={mmcls_minimum_version}, <{mmcls_maximum_version}.'

__all__ = ['__version__']
