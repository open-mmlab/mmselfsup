# Copyright (c) OpenMMLab. All rights reserved.
"""Collecting some commonly used type hint in mmselfsup."""
from typing import Optional, Union

from mmengine.config import ConfigDict

# Type hint of config data
ConfigType = Union[ConfigDict, dict]
OptConfigType = Optional[ConfigType]
