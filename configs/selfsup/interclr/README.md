# InterCLR

> [Delving into Inter-Image Invariance for Unsupervised Visual Representations](https://arxiv.org/abs/2008.11702)

<!-- [ALGORITHM] -->

## Abstract

Contrastive learning has recently shown immense
potential in unsupervised visual representation learning. Existing studies in this track mainly focus on intra-image invariance learning. The learning typically uses rich intraimage transformations to construct positive pairs and then
maximizes agreement using a contrastive loss. The merits
of inter-image invariance, conversely, remain much less explored. One major obstacle to exploit inter-image invariance
is that it is unclear how to reliably construct inter-image
positive pairs, and further derive effective supervision from
them since no pair annotations are available. In this work,
we present a comprehensive empirical study to better understand the role of inter-image invariance learning from three main constituting components: pseudo-label maintenance,
sampling strategy, and decision boundary design. To facilitate the study, we introduce a unified and generic framework that supports the integration of unsupervised intra- and
inter-image invariance learning. Through carefully-designed
comparisons and analysis, multiple valuable observations
are revealed: 1) online labels converge faster and perform
better than offline labels; 2) semi-hard negative samples are more reliable and unbiased than hard negative samples; 3) a
less stringent decision boundary is more favorable for interimage invariance learning. With all the obtained recipes, our final model, namely InterCLR, shows consistent improvements over state-of-the-art intra-image invariance learning methods on multiple standard benchmarks. We hope this
work will provide useful experience for devising effective unsupervised inter-image invariance learning.

<div align="center">
<img src="https://user-images.githubusercontent.com/52497952/205854109-2385b765-e12b-4e22-b7b8-45db6292895b.png" width="700" />
</div>

## Results and Models

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models are pre-trained on ImageNet-1k dataset. Here, we use MoCov2-InterCLR as an example. More models and results are coming soon.

### Classification

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                                                                                                                                  | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64 | k=96  |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ---- | ----- |
| [interclr-moco_resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/interclr/interclr-moco_resnet50_8xb32-coslr-200e_in1k.py) | feature5   | 85.24 | 45.08 | 59.25 | 65.99 | 74.31 | 77.95 | 80.68 | 82.7 | 83.49 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_linear-8xb32-steplr-90e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/imagenet/resnet50_mhead_linear-8xb32-steplr-90e_in1k.py) for details of config.

| Self-Supervised Config                                                                                                                                                  | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- |
| [interclr-moco_resnet50_8xb32-coslr-200e](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/interclr/interclr-moco_resnet50_8xb32-coslr-200e_in1k.py) | 15.59    | 35.10    | 47.36    | 62.86    | 68.04    |

## Citation

```bibtex
@article{xie2022delving,
  title={Delving into inter-image invariance for unsupervised visual representations},
  author={Xie, Jiahao and Zhan, Xiaohang and Liu, Ziwei and Ong, Yew-Soon and Loy, Chen Change},
  journal={International Journal of Computer Vision},
  volume={130},
  number={12},
  pages={2994--3013},
  year={2022},
  publisher={Springer}
}
```
