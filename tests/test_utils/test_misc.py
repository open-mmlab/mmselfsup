# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile

from mmselfsup.utils.misc import find_latest_checkpoint


def test_find_latest_checkpoint():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = tmpdir
        latest = find_latest_checkpoint(path)
        # There are no checkpoints in the path.
        assert latest is None

        path = osp.join(tmpdir, 'none')
        latest = find_latest_checkpoint(path)
        # The path does not exist.
        assert latest is None

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(osp.join(tmpdir, 'latest.pth'), 'w') as f:
            f.write('latest')
        path = tmpdir
        latest = find_latest_checkpoint(path)
        assert latest == osp.join(tmpdir, 'latest.pth')

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(osp.join(tmpdir, 'iter_4000.pth'), 'w') as f:
            f.write('iter_4000')
        with open(osp.join(tmpdir, 'iter_8000.pth'), 'w') as f:
            f.write('iter_8000')
        path = tmpdir
        latest = find_latest_checkpoint(path)
        assert latest == osp.join(tmpdir, 'iter_8000.pth')

    with tempfile.TemporaryDirectory() as tmpdir:
        with open(osp.join(tmpdir, 'epoch_1.pth'), 'w') as f:
            f.write('epoch_1')
        with open(osp.join(tmpdir, 'epoch_2.pth'), 'w') as f:
            f.write('epoch_2')
        path = tmpdir
        latest = find_latest_checkpoint(path)
        assert latest == osp.join(tmpdir, 'epoch_2.pth')
