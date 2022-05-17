# Copyright (c) OpenMMLab. All rights reserved.
import logging

from mmengine.logging import MMLogger


def get_root_logger(log_file: str = None,
                    log_level: int = logging.INFO) -> logging.Logger:
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to :obj:`logging.INFO`.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    try:
        return MMLogger.get_instance(
            'mmselfsup',
            logger_name='mmselfsup',
            log_file=log_file,
            log_level=log_level)
    except AssertionError:
        # if root logger already existed, no extra kwargs needed.
        return MMLogger.get_instance('mmselfsup')
