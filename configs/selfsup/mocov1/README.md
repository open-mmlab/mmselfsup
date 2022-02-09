# MoCo v1

> [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722)

<!-- [ALGORITHM] -->

## Abstract

We present Momentum Contrast (MoCo) for unsupervised visual representation learning. From a perspective on contrastive learning as dictionary look-up, we build a dynamic dictionary with a queue and a moving-averaged encoder. This enables building a large and consistent dictionary on-the-fly that facilitates contrastive unsupervised learning. MoCo provides competitive results under the common linear protocol on ImageNet classification. More importantly, the representations learned by MoCo transfer well to downstream tasks.

<div align="center">
<img  src="https://user-images.githubusercontent.com/36138628/149719892-1b6928e1-37cb-4cee-b053-ff12e1aa43c0.png" width="400" />
</div>

## Citation

```bibtex
@inproceedings{he2020momentum,
  title={Momentum contrast for unsupervised visual representation learning},
  author={He, Kaiming and Fan, Haoqi and Wu, Yuxin and Xie, Saining and Girshick, Ross},
  booktitle={CVPR},
  year={2020}
}
```
