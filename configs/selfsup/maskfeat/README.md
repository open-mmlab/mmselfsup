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

| Backbone | Pre-train epoch | Fine-tuning Top-1 |                                                            Pre-train Config                                                            |                                                                     Fine-tuning Config                                                                      |                                                                                                                            Download                                                                                                                             |
| :------: | :-------------: | :---------------: | :------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ViT-B/16 |       300       |       83.5        | [config](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/maskfeat/maskfeat_vit-base-p16_8xb256-coslr-300e_in1k.py) | [config](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/imagenet/maskfeat_vit-base-p16_ft-8xb512-coslr-100e_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/maskfeat/maskfeat_vit-base-p16_8xb256-coslr-300e_in1k_20220913-591d4c4b.pth) \| [log](https://download.openmmlab.com/mmselfsup/maskfeat/maskfeat_vit-base-p16_8xb256-coslr-300e_in1k_20220829_225552.log.json) |

## Citation

```bibtex
@inproceedings{wei2022masked,
  title={Masked feature prediction for self-supervised visual pre-training},
  author={Wei, Chen and Fan, Haoqi and Xie, Saining and Wu, Chao-Yuan and Yuille, Alan and Feichtenhofer, Christoph},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14668--14678},
  year={2022}
}
```
