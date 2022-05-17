# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

from mmengine.logging import MMLogger

from mmselfsup.utils import get_root_logger


def test_get_root_logger():
    # set all logger instance
    MMLogger._instance_dict = {}
    with tempfile.TemporaryDirectory() as tmpdirname:
        log_path = osp.join(tmpdirname, 'test.log')

        logger = get_root_logger(log_file=log_path)
        message1 = 'adhsuadghj'
        logger.info(message1)

        logger2 = get_root_logger()
        message2 = 'm,tkrgmkr'
        logger2.info(message2)

        with open(log_path, 'r') as f:
            lines = f.readlines()
            assert message1 in lines[0]
            assert message2 in lines[1]

        assert logger is logger2

        handlers = list(logger.handlers)
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
