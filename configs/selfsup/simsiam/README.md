# SimSiam

## Exploring Simple Siamese Representation Learning

<!-- [ABSTRACT] -->

Siamese networks have become a common structure in various recent models for unsupervised visual representation learning. These models maximize the similarity between two augmentations of one image, subject to certain conditions for avoiding collapsing solutions. In this paper, we report surprising empirical results that simple Siamese networks can learn meaningful representations even using none of the following: (i) negative sample pairs, (ii) large batches, (iii) momentum encoders. Our experiments show that collapsing solutions do exist for the loss and structure, but a stop-gradient operation plays an essential role in preventing collapsing. We provide a hypothesis on the implication of stop-gradient, and further show proof-of-concept experiments verifying it. Our “SimSiam” method achieves competitive results on ImageNet and downstream tasks. We hope this simple baseline will motivate people to rethink the roles of Siamese architectures for unsupervised representation learning.

<!-- [IMAGE] -->
<div align="center">
<img  />
</div>

## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{chen2021exploring,
  title={Exploring simple siamese representation learning},
  author={Chen, Xinlei and He, Kaiming},
  booktitle={CVPR},
  year={2021}
}
```

## Models and Benchmarks

**Back to [model_zoo.md](../../../docs/en/model_zoo.md) to download models.**

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models were trained on ImageNet1k dataset.

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are  Top-1 (%).

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                 | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| ---------------------------------------------------------------------- | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [resnet50_8xb32-coslr-100e](simsiam_resnet50_8xb32-coslr-100e_in1k.py) | feature5   | 84.21 | 39.71 | 49.65 | 62.79 | 69.97 | 74.73 | 78.30 | 81.06 | 82.44 |
| [resnet50_8xb32-coslr-200e](simsiam_resnet50_8xb32-coslr-200e_in1k.py) | feature5   | 85.20 | 39.85 | 50.44 | 63.73 | 70.93 | 75.74 | 79.42 | 82.02 | 83.44 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_8xb32-steplr-90e.py](../../benchmarks/classification/imagenet/resnet50_mhead_8xb32-steplr-90e_in1k.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [resnet50_8xb512-coslr-90e_in1k](../../benchmarks/classification/imagenet/resnet50_8xb512-coslr-90e_in1k.py) for details of config.

| Self-Supervised Config                                                 | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| ---------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | ------- |
| [resnet50_8xb32-coslr-100e](simsiam_resnet50_8xb32-coslr-100e_in1k.py) | 15.85    | 34.02    | 46.00    | 60.90    | 67.92    | 67.88   |
| [resnet50_8xb32-coslr-200e](simsiam_resnet50_8xb32-coslr-200e_in1k.py) | 15.57    | 37.21    | 47.28    | 62.21    | 69.85    | 69.80   |


### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [faster_rcnn_r50_c4_mstrain_24k_voc0712.py](../../benchmarks/mmdetection/voc0712/faster_rcnn_r50_c4_mstrain_24k_voc0712.py) for details of config.

| Self-Supervised Config                                                 | AP50  |
| ---------------------------------------------------------------------- | ----- |
| [resnet50_8xb32-coslr-100e](simsiam_resnet50_8xb32-coslr-100e_in1k.py) | 79.97 |
| [resnet50_8xb32-coslr-200e](simsiam_resnet50_8xb32-coslr-200e_in1k.py) | 79.85 |

#### COCO2017

Please refer to [mask_rcnn_r50_fpn_mstrain_1x_coco.py](../../benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x_coco.py) for details of config.

| Self-Supervised Config                                                 | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| ---------------------------------------------------------------------- | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [resnet50_8xb32-coslr-100e](simsiam_resnet50_8xb32-coslr-100e_in1k.py) | 38.3     | 57.6      | 41.7      | 34.4      | 54.8       | 36.9       |
| [resnet50_8xb32-coslr-200e](simsiam_resnet50_8xb32-coslr-200e_in1k.py) | 38.8     | 58.0      | 42.3      | 34.9      | 55.3       | 37.6       |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [fcn_r50-d8_512x512_20k_voc12aug.py](../../benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py) for details of config.

| Self-Supervised Config                                                 | mIOU  |
| ---------------------------------------------------------------------- | ----- |
| [resnet50_8xb32-coslr-100e](simsiam_resnet50_8xb32-coslr-100e_in1k.py) | 46.11 |
| [resnet50_8xb32-coslr-200e](simsiam_resnet50_8xb32-coslr-200e_in1k.py) | 46.27 |
