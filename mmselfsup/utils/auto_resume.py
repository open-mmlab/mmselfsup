import os.path as osp
from glob import glob


def find_available_ckpt(runner_work_dir, ckpt_out_dir):
    """Find the latest checkpoint in ckpt_out_dir.

    Args:
        runner_work_dir: the work dir of the runner
        ckpt_out_dir: the root dir, where checkpoint is saved
    Returns:
        None if checkpoints is not found, else the path of the
        latest checkpoint.
    """
    basename = osp.basename(runner_work_dir.rstrip(osp.sep))
    ckpt_out_dir = osp.join(ckpt_out_dir, basename)

    ckpts = glob(osp.join(ckpt_out_dir, "epoch_*.pth"))

    if len(ckpts) == 0:
        return None

    ckpt_dict = {
        int(ckpt.split(osp.sep)[-1].split("_")[-1][:-4]): ckpt
        for ckpt in ckpts
    }

    latest_ckpt = ckpt_dict[max(ckpt_dict)]

    return latest_ckpt