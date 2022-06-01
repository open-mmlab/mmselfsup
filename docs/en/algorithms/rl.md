# Relative Location

> [Unsupervised Visual Representation Learning by Context Prediction](https://arxiv.org/abs/1505.05192)

<!-- [ALGORITHM] -->

## Abstract

This work explores the use of spatial context as a source of free and plentiful supervisory signal for training a rich visual representation. Given only a large, unlabeled image collection, we extract random pairs of patches from each image and train a convolutional neural net to predict the position of the second patch relative to the first. We argue that doing well on this task requires the model to learn to recognize objects and their parts. We demonstrate that the feature representation learned using this within-image context indeed captures visual similarity across images. For example, this representation allows us to perform unsupervised visual discovery of objects like cats, people, and even birds from the Pascal VOC 2011 detection dataset. Furthermore, we show that the learned ConvNet can be used in the RCNN framework and provides a significant boost over a randomly-initialized ConvNet, resulting in state-of-the-art performance among algorithms which use only Pascal-provided training set annotations.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/149723222-76bc89e8-98bf-4ed7-b179-dfe5bc6336ba.png" width="400" />
</div>

## Results and Models

**Back to [model_zoo.md](https://github.com/open-mmlab/mmselfsup/blob/master/docs/en/model_zoo.md) to download models.**

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models are pre-trained on ImageNet-1k dataset.

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are Top-1 (%).

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                                       | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [resnet50_8xb64-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py) | feature4   | 65.52 | 20.36 | 23.12 | 30.66 | 37.02 | 42.55 | 50.00 | 55.58 | 59.28 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_linear-8xb32-steplr-90e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/imagenet/resnet50_mhead_linear-8xb32-steplr-90e_in1k.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [resnet50_linear-8xb32-steplr-100e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-steplr-100e_in1k.py) for details of config.

| Self-Supervised Config                                                                                                                                       | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | -------- | -------- | -------- | -------- | ------- |
| [resnet50_8xb64-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py) | 15.11    | 30.47    | 42.83    | 51.20    | 40.96    | 39.65   |

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_8xb32-steplr-28e_places205.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/places205/resnet50_mhead_8xb32-steplr-28e_places205.py) for details of config.

| Self-Supervised Config                                                                                                                                       | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | -------- | -------- | -------- | -------- |
| [resnet50_8xb64-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py) | 20.69    | 34.72    | 43.01    | 45.97    | 41.96    |

#### ImageNet Nearest-Neighbor Classification

The results are obtained from the features after GlobalAveragePooling. Here, k=10 to 200 indicates different number of nearest neighbors.

| Self-Supervised Config                                                                                                                                       | k=10 | k=20 | k=100 | k=200 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---- | ---- | ----- | ----- |
| [resnet50_8xb64-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py) | 14.5 | 15.0 | 15.0  | 14.2  |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [faster_rcnn_r50_c4_mstrain_24k_voc0712.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/voc0712/faster_rcnn_r50_c4_mstrain_24k_voc0712.py) for details of config.

| Self-Supervised Config                                                                                                                                       | AP50  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| [resnet50_8xb64-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py) | 79.70 |

#### COCO2017

Please refer to [mask_rcnn_r50_fpn_mstrain_1x_coco.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x_coco.py) for details of config.

| Self-Supervised Config                                                                                                                                       | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [resnet50_8xb64-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py) | 37.5     | 56.2      | 41.3      | 33.7      | 53.3       | 36.1       |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [fcn_r50-d8_512x512_20k_voc12aug.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py) for details of config.

| Self-Supervised Config                                                                                                                                       | mIOU  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| [resnet50_8xb64-steplr-70e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py) | 63.49 |

## Citation

```bibtex
@inproceedings{doersch2015unsupervised,
  title={Unsupervised visual representation learning by context prediction},
  author={Doersch, Carl and Gupta, Abhinav and Efros, Alexei A},
  booktitle={ICCV},
  year={2015}
}
```
