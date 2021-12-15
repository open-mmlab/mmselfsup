# Model Zoo

All models and benchmarks results are recorded below.

## Pre-trained models

| Algorithm                                                         | Config                                                                                     | Download  | Remark                |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | --------- | --------------------- |
| [BYOL](../configs/selfsup/byol/README.md)                         | [config](../configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)           | [model]() | 8xb32 / Accumulate 16 |
| [DeepCLuster](../configs/selfsup/deepcluster/README.md)           | [config](../configs/selfsup/deepcluster/deepcluster_resnet50_8xb64-steplr-200e_in1k.py)    | [model]() |                       |
| [DenseCL](../configs/selfsup/densecl/README.md)                   | [config](../configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py)             | [model]() |                       |
| [MoCo v2](../configs/selfsup/moco/README.md)                      | [config](../configs/selfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                 | [model]() |                       |
| [NPID](../configs/selfsup/npid/README.md)                         | [config](../configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                  | [model]() |                       |
|                                                                   | [config](../configs/selfsup/npid/npid-ensure-neg_resnet50_8xb32-steplr-200e_in1k.py)       | [model]() | Ensure Negative       |
| [Online DeepCluster](../configs/selfsup/odc/README.md)            | [config](../configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py)                    | [model]() |                       |
| [Relative Location](../configs/selfsup/relative_loc/README.md)    | [config](../configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)   | [model]() |                       |
| [Rotation Prediction](../configs/selfsup/rotation_pred/README.md) | [config](../configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | [model]() |                       |
| [SimCLR](../configs/selfsup/simclr/README.md)                     | [config](../configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)               | [model]() |                       |
| [SimSiam](../configs/selfsup/simsiam/README.md)                   | [config](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)             | [model]() | Epoch 100             |
|                                                                   | [config](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)             | [model]() | Epoch 200             |
| [SwAV](../configs/selfsup/swav/README.md)                         | [config](../configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py)  | [model]() |                       |

## Benchmarks

In following tables, we only displayed ImageNet Linear Evaluation, COCO17 Object Detection and PASCAL VOC12 Aug Segmentation, you can click the model name above to get the comprehensive benchmark results.

### ImageNet Linear Evaluation

If not specified, we use linear evaluation setting from [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf). Or the settings is mentioned in Remark.

| Algorithm           | Config                                                                                     | Download  | Remark                            | Top-1 (%) |
| ------------------- | ------------------------------------------------------------------------------------------ | --------- | --------------------------------- | --------- |
| BYOL                | [config](../configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)           | [model]() | 8xb32 / Accumulate 16             | 67.68     |  |
| DeepCLuster         | [config](../configs/selfsup/deepcluster/deepcluster_resnet50_8xb64-steplr-200e_in1k.py)    | [model]() |                                   |           |
| DenseCL             | [config](../configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py)             | [model]() |                                   |           |
| MoCo v2             | [config](../configs/selfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                 | [model]() |                                   | 67.56     |
| NPID                | [config](../configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                  | [model]() |                                   |           |
|                     | [config](../configs/selfsup/npid/npid-ensure-neg_resnet50_8xb32-steplr-200e_in1k.py)       | [model]() | Ensure Negative                   |           |
| Online DeepCluster  | [config](../configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py)                    | [model]() |                                   |           |
| Relative Location   | [config](../configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)   | [model]() |                                   | 39.65     |
| Rotation Prediction | [config](../configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | [model]() |                                   | 44.35     |
| SimCLR              | [config](../configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)               | [model]() |                                   |           |
| SimSiam             | [config](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)             | [model]() | Epoch 100 / SimSiam paper setting |           |
|                     | [config](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)             | [model]() | Epoch 200 / SimSiam paper setting |           |
| SwAV                | [config](../configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py)  | [model]() | SwAV paper setting                |           |

### COCO17 Object Detection

In COCO17 Object detection task, we choose the evluation protocol from [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf), with Mask-RCNN architecture, the results below are trained with the same [config](../configs/benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x.py).

| Algorithm           | Config                                                                                     | Download  | Remark                | AP (Box) | AP (Mask) |
| ------------------- | ------------------------------------------------------------------------------------------ | --------- | --------------------- | -------- | --------- |
| BYOL                | [config](../configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)           | [model]() | 8xb32 / Accumulate 16 |          |
| DeepCLuster         | [config](../configs/selfsup/deepcluster/deepcluster_resnet50_8xb64-steplr-200e_in1k.py)    | [model]() |                       |          |
| DenseCL             | [config](../configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py)             | [model]() |                       |          |
| MoCo v2             | [config](../configs/selfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                 | [model]() |                       |          |
| NPID                | [config](../configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                  | [model]() |                       |          |
|                     | [config](../configs/selfsup/npid/npid-ensure-neg_resnet50_8xb32-steplr-200e_in1k.py)       | [model]() | Ensure Negative       |          |
| Online DeepCluster  | [config](../configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py)                    | [model]() |                       |          |
| Relative Location   | [config](../configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)   | [model]() |                       |          |
| Rotation Prediction | [config](../configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | [model]() |                       |          |
| SimCLR              | [config](../configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)               | [model]() |                       |          |
| SimSiam             | [config](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)             | [model]() | Epoch 100             | 36.9     | 33.2      |
|                     | [config](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)             | [model]() | Epoch 200             |          |
| SwAV                | [config](../configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py)  | [model]() |                       |          |

### Pascal VOC12 Aug Segmentation

In Pascal VOC12 Aug Segmentation task, we choose the evluation protocol from [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf), with FCN architecture, the results below are trained with the same [config](../configs/benchmarks/mmsegmentation/voc12aug/fcn_d6_r50-d16_512x512_30k_moco.py).

| Algorithm           | Config                                                                                     | Download  | Remark                | mIOU |
| ------------------- | ------------------------------------------------------------------------------------------ | --------- | --------------------- | ---- |
| BYOL                | [config](../configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)           | [model]() | 8xb32 / Accumulate 16 |      |
| DeepCLuster         | [config](../configs/selfsup/deepcluster/deepcluster_resnet50_8xb64-steplr-200e_in1k.py)    | [model]() |                       |      |
| DenseCL             | [config](../configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py)             | [model]() |                       |      |
| MoCo v2             | [config](../configs/selfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                 | [model]() |                       |      |
| NPID                | [config](../configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                  | [model]() |                       |      |
|                     | [config](../configs/selfsup/npid/npid-ensure-neg_resnet50_8xb32-steplr-200e_in1k.py)       | [model]() | Ensure Negative       |      |
| Online DeepCluster  | [config](../configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py)                    | [model]() |                       |      |
| Relative Location   | [config](../configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)   | [model]() |                       |      |
| Rotation Prediction | [config](../configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | [model]() |                       |      |
| SimCLR              | [config](../configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)               | [model]() |                       |      |
| SimSiam             | [config](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)             | [model]() | Epoch 100             |      |
|                     | [config](../configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)             | [model]() | Epoch 200             |      |
| SwAV                | [config](../configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py)  | [model]() |                       |      |
