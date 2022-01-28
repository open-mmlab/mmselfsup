# MoCo v3

> [An Empirical Study of Training Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.02057)

<!-- [ALGORITHM] -->

## Abstract

This paper does not describe a novel method. Instead, it studies a straightforward, incremental, yet must-know baseline given the recent progress in computer vision: self-supervised learning for Vision Transformers (ViT). While the training recipes for standard convolutional networks have been highly mature and robust, the recipes for ViT are yet to be built, especially in the self-supervised scenarios where training becomes more challenging. In this work, we go back to basics and investigate the effects of several fundamental components for training self-supervised ViT. We observe that instability is a major issue that degrades accuracy, and it can be hidden by apparently good results. We reveal that these results are indeed partial failure, and they can be improved when training is made more stable. We benchmark ViT results in MoCo v3 and several other self-supervised frameworks, with ablations in various aspects. We discuss the currently positive evidence as well as challenges and open questions. We hope that this work will provide useful data points and experience for future research.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/151305362-e6e8ea35-b3b8-45f6-8819-634e67083218.png" width="500" />
</div>

## Results and Models

**Back to [model_zoo.md](../../../docs/en/model_zoo.md) to download models.**

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models were trained on ImageNet1k dataset.

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are  Top-1 (%).

#### ImageNet Linear Evaluation

The **Linear Evaluation** result is obtained by training a linear head upon the pre-trained backbone. Please refer to [vit-small-p16_8xb128-coslr-90e_in1k](../../benchmarks/classification/imagenet/vit-small-p16_8xb128-coslr-90e_in1k.py) for details of config.

| Self-Supervised Config                                                                                            | Linear Evaluation |
| ----------------------------------------------------------------------------------------------------------------- | ----------------- |
| [mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224](mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224.py) | 73.07             |

## Citation

```bibtex
@Article{chen2021mocov3,
  author  = {Xinlei Chen* and Saining Xie* and Kaiming He},
  title   = {An Empirical Study of Training Self-Supervised Vision Transformers},
  journal = {arXiv preprint arXiv:2104.02057},
  year    = {2021},
}
```
