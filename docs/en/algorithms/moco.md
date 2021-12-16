# MoCo v1 / v2

## Momentum Contrast for Unsupervised Visual Representation Learning (MoCo v1)

<!-- [ABSTRACT] -->

We present Momentum Contrast (MoCo) for unsupervised visual representation learning. From a perspective on contrastive learning as dictionary look-up, we build a dynamic dictionary with a queue and a moving-averaged encoder. This enables building a large and consistent dictionary on-the-fly that facilitates contrastive unsupervised learning. MoCo provides competitive results under the common linear protocol on ImageNet classification. More importantly, the representations learned by MoCo transfer well to downstream tasks.

<!-- [IMAGE] -->
<div align="center">
<img  />
</div>

## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{he2020momentum,
  title={Momentum contrast for unsupervised visual representation learning},
  author={He, Kaiming and Fan, Haoqi and Wu, Yuxin and Xie, Saining and Girshick, Ross},
  booktitle={CVPR},
  year={2020}
}
```

## Improved Baselines with Momentum Contrastive Learning (MoCo v2)

<!-- [ABSTRACT] -->

Contrastive unsupervised learning has recently shown encouraging progress, e.g., in Momentum Contrast (MoCo) and SimCLR. In this note, we verify the effectiveness of two of SimCLR’s design improvements by implementing them in the MoCo framework. With simple modifications to MoCo—namely, using an MLP projection head and more data augmentation—we establish stronger baselines that outperform SimCLR and do not require large training batches. We hope this will make state-of-the-art unsupervised learning research more accessible.

<!-- [IMAGE] -->
<div align="center">
<img  />
</div>

## Citation

<!-- [ALGORITHM] -->

```bibtex
@article{chen2020improved,
  title={Improved baselines with momentum contrastive learning},
  author={Chen, Xinlei and Fan, Haoqi and Girshick, Ross and He, Kaiming},
  journal={arXiv preprint arXiv:2003.04297},
  year={2020}
}
```

## Models and Benchmarks

[Back to model_zoo.md](../../../docs/model_zoo.md)

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models were trained on ImageNet1k dataset.


### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Model     | Config                                                                       | Best Layer | SVM   | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | k=96 |
| --------- | ---------------------------------------------------------------------------- | ---------- | ----- | --- | --- | --- | --- | ---- | ---- | ---- | ---- |
| [model]() | [mocov2_resnet50_8xb32-coslr-200e](mocov2_resnet50_8xb32-coslr-200e_in1k.py) | feature5   | 84.04 |     |     |     |     |      |      |      |      |

### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_8xb32-steplr-90e.py](../../benchmarks/classification/imagenet/resnet50_mhead_8xb32-steplr-90e_in1k.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [file name]() for details of config.

| Model     | Config                                                                       | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| --------- | ---------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | ------- |
| [model]() | [mocov2_resnet50_8xb32-coslr-200e](mocov2_resnet50_8xb32-coslr-200e_in1k.py) | 15.96    | 34.22    | 45.78    | 61.11    | 66.24    | 67.56   |

### iNaturalist2018 Linear Evaluation

Please refer to [resnet50_mhead_8xb32-steplr-84e_inat18.py](../../benchmarks/classification/inaturalist2018/resnet50_mhead_8xb32-steplr-84e_inat18.py) and [file name]() for details of config.

| Model     | Config                                                                       | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| --------- | ---------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | ------- |
| [model]() | [mocov2_resnet50_8xb32-coslr-200e](mocov2_resnet50_8xb32-coslr-200e_in1k.py) |          |          |          |          |          |         |

### Places205 Linear Evaluation

Please refer to [resnet50_mhead_8xb32-steplr-28e_places205.py](../../benchmarks/classification/inaturalist2018/resnet50_mhead_8xb32-steplr-28e_places205.py) and [file name]() for details of config.

| Model     | Config                                                                       | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| --------- | ---------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | ------- |
| [model]() | [mocov2_resnet50_8xb32-coslr-200e](mocov2_resnet50_8xb32-coslr-200e_in1k.py) |          |          |          |          |          |         |

#### Semi-Supervised Classification

- In this benchmark, the necks or heads are removed and only the backbone CNN is evaluated by appending a linear classification head. All parameters are fine-tuned.
- When training with 1% ImageNet, we find hyper-parameters especially the learning rate greatly influence the performance. Hence, we prepare a list of settings with the base learning rate from `{0.001, 0.01, 0.1}` and the learning rate multiplier for the head from `{1, 10, 100}`. We choose the best performing setting for each method. The setting of parameters are indicated in the file name. The learning rate is indicated like `1e-1`, `1e-2`, `1e-3` and the learning rate multiplier is indicated like `head1`, `head10`, `head100`.
- Please use --deterministic in this benchmark.

Please refer to the directories `configs/benchmarks/classification/imagenet/imagenet_1percent/` of 1% data and `configs/benchmarks/classification/imagenet/imagenet_10percent/` 10% data for details.

| Model     | Pretrain Config                                                              | Fine-tuned Config                                                                                                                                                   | Top-1 (%) | Top-5 (%) |
| --------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | --------- |
| [model]() | [mocov2_resnet50_8xb32-coslr-200e](mocov2_resnet50_8xb32-coslr-200e_in1k.py) | [resnet50_head100_4xb64-steplr1e-2-20e_in1k-1pct.py](../../benchmarks/classification/imagenet/imagenet_1percent/resnet50_head100_4xb64-steplr1e-2-20e_in1k-1pct.py) | 38.97     | 67.69     |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [faster_rcnn_r50_c4_mstrain_24k.py](../../benchmarks/mmdetection/voc0712/faster_rcnn_r50_c4_mstrain_24k.py) for details of config.

| Model     | Config                                                                       | mAP | AP50 |
| --------- | ---------------------------------------------------------------------------- | --- | ---- |
| [model]() | [mocov2_resnet50_8xb32-coslr-200e](mocov2_resnet50_8xb32-coslr-200e_in1k.py) |     |      |

#### COCO2017

Please refer to [mask_rcnn_r50_fpn_mstrain_1x.py](../../benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x.py) for details of config.

| Model     | Config                                                                       | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| --------- | ---------------------------------------------------------------------------- | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [model]() | [mocov2_resnet50_8xb32-coslr-200e](mocov2_resnet50_8xb32-coslr-200e_in1k.py) |          |           |           |           |            |            |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [file]() for details of config.

| Model     | Config                                                                       | mIOU |
| --------- | ---------------------------------------------------------------------------- | ---- |
| [model]() | [mocov2_resnet50_8xb32-coslr-200e](mocov2_resnet50_8xb32-coslr-200e_in1k.py) |      |


#### Cityscapes

Please refer to [file]() for details of config.

| Model     | Config                                                                       | mIOU |
| --------- | ---------------------------------------------------------------------------- | ---- |
| [model]() | [mocov2_resnet50_8xb32-coslr-200e](mocov2_resnet50_8xb32-coslr-200e_in1k.py) |      |
