# Model Zoo

All models and benchmarks results are recorded below.

## Pre-trained models

| Algorithm                                                         | Config                                                                                                                                   | Download                                                                                                                           |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| [BYOL](../configs/selfsup/byol/README.md)                         | [byol_resnet50_8xb32-accum16-coslr-200e_in1k](../configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)                    | [model](https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k_20211213-30dbaef1.pth)           |
|                                                                   | [byol_resnet50_8xb32-accum16-coslr-300e_in1k](../configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-300e_in1k.py)                    | [model](https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_8xb32-accum16-coslr-300e_in1k_20211213-47673e22.pth)           |
| [DeepCLuster](../configs/selfsup/deepcluster/README.md)           | [deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k](../configs/selfsup/deepcluster/deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/deepcluster/deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k-bb8681e2.pth)       |
| [DenseCL](../configs/selfsup/densecl/README.md)                   | [densecl_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py)                           | [model](https://download.openmmlab.com/mmselfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k_20211214-1efb342c.pth)             |
| [MoCo v2](../configs/selfsup/moco/README.md)                      | [mocov2_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                                | [model](https://download.openmmlab.com/mmselfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k_20211213-7ce8f840.pth)                 |
| [NPID](../configs/selfsup/npid/README.md)                         | [npid_resnet50_8xb32-steplr-200e_in1k](../configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                                  | [model](https://download.openmmlab.com/mmselfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k_20211213-b5fec6df.pth)                  |
|                                                                   | [npid-ensure-neg_resnet50_8xb32-steplr-200e_in1k](../configs/selfsup/npid/npid-ensure-neg_resnet50_8xb32-steplr-200e_in1k.py)            | [model](https://download.openmmlab.com/mmselfsup/npid/npid-ensure-neg_resnet50_8xb32-steplr-200e_in1k_20211213-1052e779.pth)       |
| [Online DeepCluster](../configs/selfsup/odc/README.md)            | [odc_resnet50_8xb64-steplr-440e_in1k](../configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py)                                     | [model](https://download.openmmlab.com/mmselfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k-5af5dd0c.pth)                             |
| [Relative Location](../configs/selfsup/relative_loc/README.md)    | [relative-loc_resnet50_8xb64-steplr-70e_in1k](../configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)            | [model](https://download.openmmlab.com/mmselfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k_20211213-cdd3162f.pth)   |
| [Rotation Prediction](../configs/selfsup/rotation_pred/README.md) | [rotation-pred_resnet50_8xb16-steplr-70e_in1k](../configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py)         | [model](https://download.openmmlab.com/mmselfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k_20211213-513972ac.pth) |
| [SimCLR](../configs/selfsup/simclr/README.md)                     | [simclr_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)                              | [model](https://download.openmmlab.com/mmselfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_20211213-d0e53669.pth)               |
| [SimSiam](../configs/selfsup/simsiam/README.md)                   | [simsiam_resnet50_8xb32-coslr-100e_in1k](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)                           | [model](https://download.openmmlab.com/mmselfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k_20211213-925d628c.pth)             |
|                                                                   | [simsiam_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)                           | [model](https://download.openmmlab.com/mmselfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k_20211213-b605f9f1.pth)             |
| [SwAV](../configs/selfsup/swav/README.md)                         | [swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](../configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py)  | [model](https://download.openmmlab.com/mmselfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96_20211213-0028900c.pth)  |

Remarks:

- If not specified, the models are trained 200 epochs.
## Benchmarks

In following tables, we only displayed ImageNet Linear Evaluation, COCO17 Object Detection and PASCAL VOC12 Aug Segmentation, you can click the model name above to get the comprehensive benchmark results.

### ImageNet Linear Evaluation

If not specified, we use linear evaluation setting from [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf). Or the settings is mentioned in Remark.

| Algorithm           | Config                                                                                                                                      | Remarks               | Top-1 (%) |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | --------- |
| BYOL                | [byol_resnet50_8xb32-accum16-coslr-200e_in1k](../configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)                       |                       | 67.68     |
| DeepCLuster         | [deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py](../configs/selfsup/deepcluster/deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py) |                       | 46.92     |
| DenseCL             | [densecl_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py)                              |                       |           |
| MoCo v2             | [mocov2_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                                   |                       | 67.56     |
| NPID                | [npid_resnet50_8xb32-steplr-200e_in1k](../configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                                     |                       |           |
|                     | [npid-ensure-neg_resnet50_8xb32-steplr-200e_in1k](../configs/selfsup/npid/npid-ensure-neg_resnet50_8xb32-steplr-200e_in1k.py)               |                       |           |
| ODC                 | [odc_resnet50_8xb64-steplr-440e_in1k](../configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py)                                        |                       | 53.42     |
| Relative Location   | [relative-loc_resnet50_8xb64-steplr-70e_in1k](../configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)               |                       | 39.65     |
| Rotation Prediction | [rotation-pred_resnet50_8xb16-steplr-70e_in1k](../configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py)            |                       | 44.35     |
| SimCLR              | [simclr_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)                                 |                       | 58.92     |
| SimSiam             | [simsiam_resnet50_8xb32-coslr-100e_in1k](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)                              | SimSiam paper setting | 67.88     |
|                     | [simsiam_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)                              | SimSiam paper setting | 69.80     |
| SwAV                | [swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](../configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py)     | SwAV paper setting    | 70.55     |

### COCO17 Object Detection

In COCO17 Object detection task, we choose the evluation protocol from [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf), with Mask-RCNN architecture, the results below are trained with the same [config](../configs/benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x.py).

| Algorithm           | Config                                                                                                                                   | mAP (Box) | mAP (Mask) |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---------- |
| BYOL                | [byol_resnet50_8xb32-accum16-coslr-200e_in1k](../configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)                    |           |            |
| DeepCLuster         | [deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k](../configs/selfsup/deepcluster/deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py) |           |            |
| DenseCL             | [densecl_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py)                           |           |            |
| MoCo v2             | [mocov2_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                                |           |            |
| NPID                | [npid_resnet50_8xb32-steplr-200e_in1k](../configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                                  |           |            |
|                     | [npid-ensure-neg_resnet50_8xb32-steplr-200e_in1k](../configs/selfsup/npid/npid-ensure-neg_resnet50_8xb32-steplr-200e_in1k.py)            |           |            |
| ODC                 | [odc_resnet50_8xb64-steplr-440e_in1k](../configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py)                                     |           |            |
| Relative Location   | [relative-loc_resnet50_8xb64-steplr-70e_in1k](../configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)            |           |            |
| Rotation Prediction | [rotation-pred_resnet50_8xb16-steplr-70e_in1k](../configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py)         |           |            |
| SimCLR              | [simclr_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)                              |           |            |
| SimSiam             | [simsiam_resnet50_8xb32-coslr-100e_in1k](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)                           |           |            |
|                     | [simsiam_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)                           |           |            |
| SwAV                | [swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](../configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py)  |           |            |

### Pascal VOC12 Aug Segmentation

In Pascal VOC12 Aug Segmentation task, we choose the evluation protocol from [MMSeg](https://github.com/open-mmlab/mmsegmentation), with FCN architecture, the results below are trained with the same [config](configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py).

| Algorithm           | Config                                                                                                                                   | mIOU  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| BYOL                | [byol_resnet50_8xb32-accum16-coslr-200e_in1k](../configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)                    | 67.16 |
| DeepCLuster         | [deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k](../configs/selfsup/deepcluster/deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py) | 59.69 |
| DenseCL             | [densecl_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py)                           | 69.47 |
| MoCo v2             | [mocov2_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                                | 67.55 |
| NPID                | [npid_resnet50_8xb32-steplr-200e_in1k](../configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                                  | 65.45 |
|                     | [npid-ensure-neg_resnet50_8xb32-steplr-200e_in1k](../configs/selfsup/npid/npid-ensure-neg_resnet50_8xb32-steplr-200e_in1k.py)            | 64.73 |
| ODC                 | [odc_resnet50_8xb64-steplr-440e_in1k](../configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py)                                     | 54.76 |
| Relative Location   | [relative-loc_resnet50_8xb64-steplr-70e_in1k](../configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)            | 63.49 |
| Rotation Prediction | [rotation-pred_resnet50_8xb16-steplr-70e_in1k](../configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py)         | 64.31 |
| SimCLR              | [simclr_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)                              | 64.03 |
| SimSiam             | [simsiam_resnet50_8xb32-coslr-100e_in1k](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)                           | 46.11 |
|                     | [simsiam_resnet50_8xb32-coslr-200e_in1k](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)                           | 46.27 |
| SwAV                | [swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](../configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py)  | 63.73 |
