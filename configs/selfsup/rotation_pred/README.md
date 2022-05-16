# Rotation Prediction

> [Unsupervised Representation Learning by Predicting Image Rotation](https://arxiv.org/abs/1803.07728)

<!-- [ALGORITHM] -->

## Abstract

Over the last years, deep convolutional neural networks (ConvNets) have transformed the field of computer vision thanks to their unparalleled capacity to learn high level semantic image features. However, in order to successfully learn those features, they usually require massive amounts of manually labeled data, which is both expensive and impractical to scale. Therefore, unsupervised semantic feature learning, i.e., learning without requiring manual annotation effort, is of crucial importance in order to successfully harvest the vast amount of visual data that are available today. In our work we propose to learn image features by training ConvNets to recognize the 2d rotation that is applied to the image that it gets as input. We demonstrate both qualitatively and quantitatively that this apparently simple task actually provides a very powerful supervisory signal for semantic feature learning. We exhaustively evaluate our method in various unsupervised feature learning benchmarks and we exhibit in all of them state-of-the-art performance. Specifically, our results on those benchmarks demonstrate dramatic improvements w.r.t. prior state-of-the-art approaches in unsupervised representation learning and thus significantly close the gap with supervised feature learning.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/149723477-8f63e237-362e-4962-b405-9bab0f579808.png" width="700" />
</div>

## Results and Models

**Back to [model_zoo.md](https://github.com/open-mmlab/mmselfsup/blob/master/docs/en/model_zoo.md) to download models.**

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models are pre-trained on ImageNet-1k dataset.

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are Top-1 (%).

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                                         | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [resnet50_8xb16-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | feature4   | 67.70 | 20.60 | 24.35 | 31.41 | 39.17 | 46.56 | 53.37 | 59.14 | 62.42 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_linear-8xb32-steplr-90e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/imagenet/resnet50_mhead_linear-8xb32-steplr-90e_in1k.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [resnet50_linear-8xb32-steplr-100e_in1k.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-steplr-100e_in1k.py) for details of config.

| Self-Supervised Config                                                                                                                                         | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | ------- |
| [resnet50_8xb16-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | 12.15    | 31.99    | 44.57    | 54.20    | 45.94    | 48.12   |

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_8xb32-steplr-28e_places205.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/places205/resnet50_mhead_8xb32-steplr-28e_places205.py) for details of config.

| Self-Supervised Config                                                                                                                                         | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [resnet50_8xb16-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | 18.94    | 34.72    | 44.53    | 46.30    | 44.12    |

#### ImageNet Nearest-Neighbor Classification

The results are obtained from the features after GlobalAveragePooling. Here, k=10 to 200 indicates different number of nearest neighbors.

| Self-Supervised Config                                                                                                                                         | k=10 | k=20 | k=100 | k=200 |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---- | ---- | ----- | ----- |
| [resnet50_8xb16-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | 11.0 | 11.9 | 12.6  | 12.4  |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [faster_rcnn_r50_c4_mstrain_24k_voc0712.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/voc0712/faster_rcnn_r50_c4_mstrain_24k_voc0712.py) for details of config.

| Self-Supervised Config                                                                                                                                         | AP50  |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [resnet50_8xb16-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | 79.67 |

#### COCO2017

Please refer to [mask_rcnn_r50_fpn_mstrain_1x_coco.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x_coco.py) for details of config.

| Self-Supervised Config                                                                                                                                         | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [resnet50_8xb16-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | 37.9     | 56.5      | 41.5      | 34.2      | 53.9       | 36.7       |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [fcn_r50-d8_512x512_20k_voc12aug.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py) for details of config.

| Self-Supervised Config                                                                                                                                         | mIOU  |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| [resnet50_8xb16-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py) | 64.31 |

## Citation

```bibtex
@inproceedings{komodakis2018unsupervised,
  title={Unsupervised representation learning by predicting image rotations},
  author={Komodakis, Nikos and Gidaris, Spyros},
  booktitle={ICLR},
  year={2018}
}
```
