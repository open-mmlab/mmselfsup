# MaskFeat

> [Masked Feature Prediction for Self-Supervised Visual Pre-Training](https://arxiv.org/abs/2112.09133v1)

<!-- [ALGORITHM] -->

## Abstract

We present Masked Feature Prediction (MaskFeat) for self-supervised pre-training of video models. Our approach first randomly masks out a portion of the input sequence and then predicts the feature of the masked regions. We study five different types of features and find Histograms of Oriented Gradients (HOG), a hand-crafted feature descriptor, works particularly well in terms of both performance and efficiency. We observe that the local contrast normalization in HOG is essential for good results, which is in line with earlier work using HOG for visual recognition. Our approach can learn abundant visual knowledge and drive large-scale Transformer-based models. Without using extra model weights or supervision, MaskFeat pre-trained on unlabeled videos achieves unprecedented results of 86.7% with MViT-L on Kinetics-400, 88.3% on Kinetics-600, 80.4% on Kinetics-700, 38.8 mAP on AVA, and 75.0% on SSv2. MaskFeat further generalizes to image input, which can be interpreted as a video with a single frame and obtains competitive results on ImageNet.

<div align="center">
<img src="https://user-images.githubusercontent.com/48178838/190090285-428f07c0-0887-4ce8-b94f-f719cfd25622.png" width="60%"/>
</div>

## Models and Benchmarks

Here, we report the results of the model, which is pre-trained on ImageNet-1k
for 400 epochs, the details are below:

| Backbone | Pre-train epoch | Fine-tuning Top-1 |                                                            Pre-train Config                                                            |                                                                     Fine-tuning Config                                                                      |                                                                                                                      Download                                                                                                                       |
| :------: | :-------------: | :---------------: | :------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ViT-B/16 |       300       |       83.5        | [config](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/maskfeat/maskfeat_vit-base-p16_8xb256-coslr-300e_in1k.py) | [config](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/imagenet/maskfeat_vit-base-p16_ft-8xb512-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k-224_20220223-85be947b.pth) \| [log](https://download.openmmlab.com/mmselfsup/mae/mae_vit-base-p16_8xb512-coslr-300e_in1k-224_20220210_140925.log.json) |

## Citation

```bibtex
@article{He2021MaskedAA,
  title={Masked Autoencoders Are Scalable Vision Learners},
  author={Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and
  Piotr Doll'ar and Ross B. Girshick},
  journal={ArXiv},
  year={2021}
}
```
