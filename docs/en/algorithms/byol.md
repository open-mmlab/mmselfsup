# BYOL

> [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)

<!-- [ALGORITHM] -->

## Abstract

**B**ootstrap **Y**our **O**wn **L**atent (BYOL) is a new approach to self-supervised image representation learning. BYOL relies on two neural networks, referred to as online and target networks, that interact and learn from each other. From an augmented view of an image, we train the online network to predict the target network representation of the same image under a different augmented view. At the same time, we update the target network with a slow-moving average of the online network.

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/149720208-5ffbee78-1437-44c7-9ddb-b8caab60d2c3.png" width="800" />
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
| [resnet50_8xb32-accum16-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py) | feature5   | 86.31 | 45.37 | 56.83 | 68.47 | 74.12 | 78.30 | 81.53 | 83.56 | 84.73 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_linear-8xb32-steplr-90e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/imagenet/resnet50_mhead_linear-8xb32-steplr-90e_in1k.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [resnet50_linear-8xb512-coslr-90e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/imagenet/resnet50_linear-8xb512-coslr-90e_in1k.py) for details of config.

| Self-Supervised Config                                                                                                                                       | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | -------- | -------- | -------- | -------- | ------- |
| [resnet50_8xb32-accum16-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py) | 15.16    | 35.26    | 47.77    | 63.10    | 71.21    | 71.72   |
| [resnet50_16xb256-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_16xb256-coslr-200e_in1k.py)             | 15.41    | 35.15    | 47.77    | 62.59    | 71.85    | 71.88   |

#### Places205 Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_8xb32-steplr-28e_places205.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/places205/resnet50_mhead_8xb32-steplr-28e_places205.py) for details of config.

| Self-Supervised Config                                                                                                                                       | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | -------- | -------- | -------- | -------- |
| [resnet50_8xb32-accum16-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py) | 21.25    | 36.55    | 43.66    | 50.74    | 53.82    |
| [resnet50_8xb32-accum16-coslr-300e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-300e_in1k.py) | 21.18    | 36.68    | 43.42    | 51.04    | 54.06    |

#### ImageNet Nearest-Neighbor Classification

The results are obtained from the features after GlobalAveragePooling. Here, k=10 to 200 indicates different number of nearest neighbors.

| Self-Supervised Config                                                                                                                                       | k=10 | k=20 | k=100 | k=200 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---- | ---- | ----- | ----- |
| [resnet50_8xb32-accum16-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py) | 63.9 | 64.2 | 62.9  | 61.9  |
| [resnet50_8xb32-accum16-coslr-300e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-300e_in1k.py) | 66.1 | 66.3 | 65.2  | 64.4  |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [faster_rcnn_r50_c4_mstrain_24k_voc0712.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/voc0712/faster_rcnn_r50_c4_mstrain_24k_voc0712.py) for details of config.

| Self-Supervised Config                                                                                                                                       | AP50  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| [resnet50_8xb32-accum16-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py) | 80.35 |

#### COCO2017

Please refer to [mask_rcnn_r50_fpn_mstrain_1x_coco.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x_coco.py) for details of config.

| Self-Supervised Config                                                                                                                                       | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [resnet50_8xb32-accum16-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py) | 40.9     | 61.0      | 44.6      | 36.8      | 58.1       | 39.5       |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [fcn_r50-d8_512x512_20k_voc12aug.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py) for details of config.

| Self-Supervised Config                                                                                                                                       | mIOU  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----- |
| [resnet50_8xb32-accum16-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py) | 67.16 |

## Citation

```bibtex
@inproceedings{grill2020bootstrap,
  title={Bootstrap your own latent: A new approach to self-supervised learning},
  author={Grill, Jean-Bastien and Strub, Florian and Altch{\'e}, Florent and Tallec, Corentin and Richemond, Pierre H and Buchatskaya, Elena and Doersch, Carl and Pires, Bernardo Avila and Guo, Zhaohan Daniel and Azar, Mohammad Gheshlaghi and others},
  booktitle={NeurIPS},
  year={2020}
}
```
