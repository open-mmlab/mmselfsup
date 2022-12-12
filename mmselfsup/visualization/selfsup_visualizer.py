# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import mmcv
import numpy as np
from mmengine.dist import master_only
from mmengine.structures import InstanceData
from mmengine.visualization import Visualizer

from mmselfsup.registry import VISUALIZERS
from mmselfsup.structures import SelfSupDataSample


@VISUALIZERS.register_module()
class SelfSupVisualizer(Visualizer):
    """MMSelfSup Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        alpha (int, float): The transparency of boxes or mask.
                Defaults to 0.8.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from mmengine.structures import InstanceData
        >>> from mmselfsup.structures import SelfSupDataSample
        >>> from mmselfsup.visualization import SelfSupVisualizer

        >>> selfsup_visualizer = SelfSupVisualizer()
        >>> image = np.random.randint(0, 256,
        ...                     size=(10, 12, 3)).astype('uint8')
        >>> pseudo_label = InstanceData()
        >>> pseudo_label.patch_box = torch.Tensor([[1, 2, 2, 5]])
        >>> gt_selfsup_data_sample = SelfSupDataSample()
        >>> gt_selfsup_data_sample.pseudo_label = pseudo_label
        >>> selfsup_visualizer.add_datasample('image', image,
        ...                         gt_selfsup_data_sample)
        >>> selfsup_visualizer.add_datasample(
        ...                       'image', image, gt_selfsup_data_sample,
        ...                        out_file='out_file.jpg')
        >>> selfsup_visualizer.add_datasample(
        ...                        'image', image, gt_selfsup_data_sample,
        ...                         show=True)
        >>> pseudo_label = InstanceData()
        >>> pseudo_label.patch_box = torch.Tensor([[1, 2, 2, 5]])
        >>> pred_selfsup_data_sample = SelfSupDataSample()
        >>> pred_selfsup_data_sample.pseudo_label = pseudo_label
        >>> selfsup_visualizer.add_datasample('image', image,
        ...                         gt_selfsup_data_sample,
        ...                         pred_selfsup_data_sample)
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[List[Dict]] = None,
                 save_dir: Optional[str] = None,
                 line_width: Union[int, float] = 3,
                 alpha: Union[int, float] = 0.8):
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir)
        self.line_width = line_width
        self.alpha = alpha
        # Set default value. When calling
        # `SelfSupVisualizer().dataset_meta=xxx`,
        # it will override the default value.
        self.dataset_meta = {}

    def _draw_boxes(
        self,
        image: np.ndarray,
        boxes: InstanceData,
        edge_colors: Union[str, tuple, List[str], List[tuple]] = 'r'
    ) -> np.ndarray:
        """Draw instance with boxes.

        Args:
            image (np.ndarray): The image to draw.
            boxes (:obj:`InstanceData`): Data structure for
                instance-level box annotations.
            edge_colors (Union[str, tuple, List[str], List[tuple]]): The
                colors of boxes. ``colors`` can have the same length with
                lines or just single value. If ``colors`` is single value, all
                the lines will have the same colors. Refer to `matplotlib.
                colors` for full list of formats that are accepted.
                Defaults to 'r'.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image.copy())
        self.draw_bboxes(
            boxes,
            edge_colors=edge_colors,
            alpha=self.alpha,
            line_widths=self.line_width)

        return self.get_image()

    def _draw_mask(
            self,
            image: np.ndarray,
            mask: InstanceData,
            colors: Union[str, tuple, List[str],
                          List[tuple]] = 'k') -> np.ndarray:
        """Draw instance with binary mask.

        Args:
            image (np.ndarray): The image to draw.
            mask (:obj:`InstanceData`): Data structure for
                pixel-level annotations.
            colors (Union[str, tuple, List[str], List[tuple]]): The colors
                which binary_masks will convert to. ``colors`` can have
                the same length with binary_masks or just single value.
                If ``colors`` is single value, all the binary_masks will
                convert to the same colors. The colors format is RGB.
                Defaults to np.array([0, 0, 0]).

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image.copy())
        if 'value' in mask:
            mask = mask.value
            mask_ = np.zeros((image.shape[0], image.shape[1]))
            num_mask = [
                image.shape[0] // mask.shape[0],
                image.shape[1] // mask.shape[1]
            ]
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    mask_[i][j] = mask[i // num_mask[0]][j // num_mask[1]]
            self.draw_binary_masks(
                mask_.astype(np.bool_), colors=colors, alphas=self.alpha)

        return self.get_image()

    @master_only
    def add_datasample(self,
                       name: str,
                       image: np.ndarray,
                       gt_sample: Optional[SelfSupDataSample] = None,
                       pred_sample: Optional[SelfSupDataSample] = None,
                       draw_gt: bool = True,
                       draw_pred: bool = True,
                       show: bool = False,
                       wait_time: float = 0,
                       out_file: Optional[str] = None,
                       step: int = 0) -> None:
        """Draw datasample and save to all backends.

          - If GT and prediction are plotted at the same time, they are
            displayed in a stitched image where the left image is the
            ground truth and the right image is the prediction.

          - If ``show`` is True, all storage backends are ignored, and
            the images will be displayed in a local window.

          - If ``out_file`` is specified, the drawn image will be
            saved to ``out_file``. t is usually used when the display
            is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            gt_sample (:obj:`SelfSupDataSample`, optional): GT
                SelfSupDataSample. Defaults to None.
            pred_sample (:obj:`SelfSupDataSample`, optional): Prediction
                SelfSupDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT SelfSupDataSample.
                Default to True.
            draw_pred (bool): Whether to draw Prediction SelfSupDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        gt_img_data = None
        pred_img_data = None

        if draw_gt and gt_sample is not None:
            gt_img_data = image
            if 'pseudo_label' in gt_sample:
                if ('patch_box' in gt_sample.pseudo_label) and \
                   ('unpatched_img' in gt_sample.pseudo_label):
                    gt_img_data = self._draw_boxes(
                        gt_sample.pseudo_label.unpatched_img[0, ::].numpy()[
                            ..., [2, 1, 0]],
                        gt_sample.pseudo_label.patch_box[0, ::].numpy())

            if 'mask' in gt_sample:
                gt_img_data = self._draw_mask(gt_img_data,
                                              gt_sample.mask.numpy())

        if draw_pred and pred_sample is not None:
            pred_img_data = image
            if 'pseudo_label' in gt_sample:
                if ('patch_box' in gt_sample.pseudo_label) and \
                   ('unpatched_img' in gt_sample.pseudo_label):
                    pred_img_data = self._draw_boxes(
                        pred_sample.pseudo_label.unpatched_img[0, ::].numpy(),
                        pred_sample.pseudo_label.patch_box[0, ::].numpy())

            if 'mask' in pred_sample:
                pred_img_data = self._draw_mask(pred_img_data,
                                                pred_sample.mask.numpy())

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)
        else:
            self.add_image(name, drawn_img, step)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
