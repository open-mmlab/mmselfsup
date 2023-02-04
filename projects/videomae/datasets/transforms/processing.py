# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import BaseTransform

from mmselfsup.registry import TRANSFORMS


@TRANSFORMS.register_module()
class VideoMAEMaskGenerator(BaseTransform):
    """Generate mask for VideoMAE."""

    def __init__(self,
                 input_size: tuple,
                 mask_ratio: float = 0.75,
                 patch_size: int = 16,
                 tubelet_size: int = 2,
                 mask_mode: str = 'tube') -> None:
        self.input_size = input_size
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode

        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        num_frames, height, width = input_size

        self.num_tubelets = num_frames // tubelet_size

        self.num_patches_frame = (height // patch_size) * (width // patch_size)
        self.num_patches_video = self.num_tubelets * self.num_patches_frame

        self.num_masks_frame = int(mask_ratio * self.num_patches_frame)
        self.num_masks_video = self.num_tubelets * self.num_patches_frame

    def transform(self, results: dict) -> dict:
        if self.mask_mode == 'random':
            # TODO: add random mask
            pass
        elif self.mask_mode == 'tube':
            mask_frame = np.hstack([
                np.zeros(self.num_patches_frame - self.num_masks_frame),
                np.ones(self.num_masks_frame)
            ])

            np.random.shuffle(mask_frame)
            mask_video = np.tile(mask_frame, (self.num_tubelets, 1)).flatten()

        else:
            raise NotImplementedError

        results.update({'mask': mask_video})

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(input_size={self.input_size}, '
        repr_str += f'mask_ratio={self.mask_ratio}, '
        repr_str += f'mask_mode={self.mask_mode}, '
        repr_str += f'patch_size={self.patch_size}, '
        repr_str += f'tubelet_size={self.tubelet_size},)'
        return repr_str
