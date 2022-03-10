# Model Zoo

All models and part of benchmark results are recorded below.

## Pre-trained models

| Algorithm                                                                                                          | Config                                                                                                                                                                                       | Download                                                                                                                                                                                                                                                                      |
| ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [BYOL](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/README.md)                         | [byol_resnet50_8xb32-accum16-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)                       | [model](https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k_20220225-5c8b2c2e.pth) &#124; [log](https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k_20220214_115709.log.json)                     |
|                                                                                                                    | [byol_resnet50_8xb32-accum16-coslr-300e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-300e_in1k.py)                       | [model](https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_8xb32-accum16-coslr-300e_in1k_20220225-a0daa54a.pth) &#124; [log](https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_8xb32-accum16-coslr-300e_in1k_20220210_095852.log.json)                     |
| [DeepCluster](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/deepcluster/README.md)           | [deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/deepcluster/deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py)    | [model](https://download.openmmlab.com/mmselfsup/deepcluster/deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k-bb8681e2.pth)                                                                                                                                                  |
| [DenseCL](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/densecl/README.md)                   | [densecl_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py)                              | [model](https://download.openmmlab.com/mmselfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k_20220225-8c7808fe.pth) &#124; [log](https://download.openmmlab.com/mmselfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k_20220215_041207.log.json)                         |
| [MoCo v2](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov2/README.md)                    | [mocov2_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                                 | [model](https://download.openmmlab.com/mmselfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k_20220225-89e03af4.pth) &#124; [log](https://download.openmmlab.com/mmselfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k_20220210_110905.log.json)                                 |
| [NPID](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/npid/README.md)                         | [npid_resnet50_8xb32-steplr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                                     | [model](https://download.openmmlab.com/mmselfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k_20220225-5fbbda2a.pth) &#124; [log](https://download.openmmlab.com/mmselfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k_20220215_185513.log.json)                                   |
| [ODC](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/odc/README.md)                           | [odc_resnet50_8xb64-steplr-440e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py)                                        | [model](https://download.openmmlab.com/mmselfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k_20220225-a755d9c0.pth)   &#124; [log](https://download.openmmlab.com/mmselfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k_20220215_235245.log.json)                                     |
| [Relative Location](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/README.md)    | [relative-loc_resnet50_8xb64-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)               | [model](https://download.openmmlab.com/mmselfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k_20220225-84784688.pth) &#124; [log](https://download.openmmlab.com/mmselfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k_20220211_124808.log.json)     |
| [Rotation Prediction](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/README.md) | [rotation-pred_resnet50_8xb16-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py)            | [model](https://download.openmmlab.com/mmselfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k_20220225-5b9f06a0.pth) &#124; [log](https://download.openmmlab.com/mmselfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k_20220215_185303.log.json) |
| [SimCLR](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simclr/README.md)                     | [simclr_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)                                 | [model](simclr_resnet50_8xb32-coslr-200e_in1k_20220225-97d2abef.pth)        &#124; [log](https://download.openmmlab.com/mmselfsup/simclr/simclr_resnet50_8xb64-coslr-200e_in1k_20220210_191629.log.json)                                                                      |
| [SimSiam](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/README.md)                   | [simsiam_resnet50_8xb32-coslr-100e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)                              | [model](https://download.openmmlab.com/mmselfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k_20220225-68a88ad8.pth) &#124; [log](https://download.openmmlab.com/mmselfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k_20220210_195405.log.json)                         |
|                                                                                                                    | [simsiam_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)                              | [model](https://download.openmmlab.com/mmselfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k_20220225-2f488143.pth) &#124; [log](https://download.openmmlab.com/mmselfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k_20220210_195402.log.json)                         |
| [SwAV](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/README.md)                         | [swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py)     | [model](https://download.openmmlab.com/mmselfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96_20220225-0497dd5d.pth) &#124; [log](https://download.openmmlab.com/mmselfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96_20220211_061131.log.json)   |
| [MoCo v3](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov3/README.md)                    | [mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov3/mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224.py) | [model](https://download.openmmlab.com/mmselfsup/moco/mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224_20220225-e31238dd.pth) &#124; [log](https://download.openmmlab.com/mmselfsup/moco/mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224_20220222_160222.log.json) |
| [MAE](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mae/README.md)                           | [mae_vit-base-p16_8xb512-coslr-400e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py)                                | [model](https://download.openmmlab.com/mmselfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k-224_20220223-85be947b.pth) &#124; [log](https://download.openmmlab.com/mmselfsup/mae/mae_vit-base-p16_8xb512-coslr-300e_in1k-224_20220210_140925.log.json)                       |

Remarks:

- The training details are recorded in the config names.

- You can click algorithm name to obtain more information.

## Benchmarks

In following tables, we only displayed ImageNet Linear Evaluation, COCO17 Object Detection and PASCAL VOC12 Aug Segmentation, you can click algorithm name above to check the comprehensive benchmark results.

### Benchmark Table

Pre-train on ImageNet-1k

Linear Probing on ImageNet-1k

Fine-tuning on ImageNet-1k

Detection on COCO 

Segmentation on ADE20K

| Method        | Backbone  | Pre-train Epoch | Linear Probing | Fine-tuning | Detection | Segmentation | Paper Link                               |
| ------------- | --------- | --------------- | -------------- | ----------- | --------- | ------------ | ---------------------------------------- |
| Relative-Loc  | ResNet 50 | 70              | 38.78*         |             |           |              | [link](https://arxiv.org/abs/1505.05192) |
| Rotation-Pred | ResNet 50 | 70              | 48.12*         |             |           |              | [link](https://arxiv.org/abs/1803.07728) |
| DeepCluster   | ResNet 50 | 200             | 46.92*         |             |           |              | [link](https://arxiv.org/abs/1807.05520) |
| NPID          | ResNet 50 | 200             | 58.97*         |             |           |              | [link](https://arxiv.org/abs/1805.01978) |
| ODC           | ResNet 50 | 440             | 53.43*         |             |           |              | [link](https://arxiv.org/abs/2006.10645) |
| SimCLR        | ResNet 50 | 200             | 57.28*         |             |           |              | [link](https://arxiv.org/abs/2002.05709) |
| MoCo v2       | ResNet 50 | 200             | 67.58          |             |           |              | [link](https://arxiv.org/abs/2003.04297) |
| BYOL          | ResNet 50 | 200             | 67.55*         |             |           |              | [link](https://arxiv.org/abs/2006.07733) |
| BYOL          | ResNet 50 | 300             | 68.55*         |             |           |              | [link](https://arxiv.org/abs/2006.07733) |
| SwAV          | ResNet 50 | 200             | 70.47*         |             | 40.2*     |              | [link](https://arxiv.org/abs/2006.09882) |
| DenseCL       | ResNet 50 | 200             | 63.62*         |             |           |              | [link](https://arxiv.org/abs/2011.09157) |
| SimSiam       | ResNet 50 | 100             | 68.28          |             |           |              | [link](https://arxiv.org/abs/2011.10566) |
| SimSiam       | ResNet 50 | 200             | 69.84          |             |           |              | [link](https://arxiv.org/abs/2011.10566) |
| MoCo v3       | ViT Small | 300             | 73.19          |             |           |              | [link](https://arxiv.org/abs/2104.02057) |
| MAE           | ViT Base  | 400             |                | 83.1        |           |              | [link](https://arxiv.org/abs/2111.06377) |

### ImageNet Linear Evaluation

If not specified, we use linear evaluation setting from [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf). Or the settings is mentioned in Remark.

| Algorithm           | Config                                                                                                                                                                                       | Remarks               | Top-1 (%) |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | --------- |
| BYOL                | [byol_resnet50_8xb32-accum16-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)                       |                       | 67.55     |
|                     | [byol_resnet50_8xb32-accum16-coslr-300e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-300e_in1k.py)                       |                       | 68.55     |
| DeepCluster         | [deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/deepcluster/deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py) |                       | 46.92     |
| DenseCL             | [densecl_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py)                              |                       | 63.62     |
| MoCo v2             | [mocov2_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                                 |                       | 67.58     |
| NPID                | [npid_resnet50_8xb32-steplr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                                     |                       | 58.97     |
| ODC                 | [odc_resnet50_8xb64-steplr-440e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py)                                        |                       | 53.43     |
| Relative Location   | [relative-loc_resnet50_8xb64-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)               |                       | 38.78     |
| Rotation Prediction | [rotation-pred_resnet50_8xb16-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py)            |                       | 48.12     |
| SimCLR              | [simclr_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)                                 |                       | 57.28     |
| SimSiam             | [simsiam_resnet50_8xb32-coslr-100e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)                              | SimSiam paper setting | 68.28     |
|                     | [simsiam_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)                              | SimSiam paper setting | 69.84     |
| SwAV                | [swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py)     | SwAV paper setting    | 70.47     |
| MoCo v3             | [mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov3/mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224.py) | MoCo v3 paper setting | 73.19     |


### ImageNet Fine-tuning
| Algorithm | Config                                                                                                                                                        | Remarks | Top-1 (%) |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------- |
| MAE       | [mae_vit-base-p16_8xb512-coslr-400e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py) |         | 83.1      |

### COCO17 Object Detection

In COCO17 Object detection task, we choose the evluation protocol from [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf), with Mask-RCNN architecture, the results below are trained with the same [config](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x_coco.py).

| Algorithm           | Config                                                                                                                                                                                   | mAP (Box) | mAP (Mask) |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---------- |
| BYOL                | [byol_resnet50_8xb32-accum16-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)                   | 40.9      | 36.8       |
| DenseCL             | [densecl_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py)                          |           |            |
| MoCo v2             | [mocov2_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                             | 40.2      | 36.1       |
| NPID                | [npid_resnet50_8xb32-steplr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                                 | 38.5      | 34.6       |
| Relative Location   | [relative-loc_resnet50_8xb64-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)           | 37.5      | 33.7       |
| Rotation Prediction | [rotation-pred_resnet50_8xb16-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py)        | 37.9      | 34.2       |
| SimCLR              | [simclr_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)                             | 38.7      | 34.9       |
| SimSiam             | [simsiam_resnet50_8xb32-coslr-100e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)                          | 38.6      | 34.6       |
|                     | [simsiam_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)                          | 38.8      | 34.9       |
| SwAV                | [swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py) | 40.2      | 36.3       |

### Pascal VOC12 Aug Segmentation

In Pascal VOC12 Aug Segmentation task, we choose the evluation protocol from [MMSeg](https://github.com/open-mmlab/mmsegmentation), with FCN architecture, the results below are trained with the same [config](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py).

| Algorithm           | Config                                                                                                                                                                                   | mIOU  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| BYOL                | [byol_resnet50_8xb32-accum16-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)                   | 67.16 |
| DenseCL             | [densecl_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py)                          | 69.47 |
| MoCo v2             | [mocov2_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                             | 67.55 |
| NPID                | [npid_resnet50_8xb32-steplr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                                 | 65.45 |
| Relative Location   | [relative-loc_resnet50_8xb64-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)           | 63.49 |
| Rotation Prediction | [rotation-pred_resnet50_8xb16-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py)        | 64.31 |
| SimCLR              | [simclr_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)                             | 64.03 |
| SimSiam             | [simsiam_resnet50_8xb32-coslr-100e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)                          | 48.35 |
|                     | [simsiam_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)                          | 46.27 |
| SwAV                | [swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py) | 63.73 |
