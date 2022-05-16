# SimMIM

> [SimMIM: A Simple Framework for Masked Image Modeling](https://arxiv.org/abs/2111.09886)

<!-- [ALGORITHM] -->

## Abstract

This paper presents SimMIM, a simple framework for masked image modeling. We simplify recently proposed related approaches without special designs such as blockwise masking and tokenization via discrete VAE or clustering. To study what let the masked image modeling task learn good representations, we systematically study the major components in our framework, and find that simple designs of each component have revealed very strong representation learning performance: 1) random masking of the input image with a moderately large masked patch size (e.g., 32) makes a strong pre-text task; 2) predicting raw pixels of RGB values by direct regression performs no worse than the patch classification approaches with complex designs; 3) the prediction head can be as light as a linear layer, with no worse performance than heavier ones. Using ViT-B, our approach achieves 83.8% top-1 fine-tuning accuracy on ImageNet-1K by pre-training also on this dataset, surpassing previous best approach by +0.6%. When applied on a larger model of about 650 million parameters, SwinV2H, it achieves 87.1% top-1 accuracy on ImageNet-1K using only ImageNet-1K data. We also leverage this approach to facilitate the training of a 3B model (SwinV2-G), that by 40× less data than that in previous practice, we achieve the state-of-the-art on four representative vision benchmarks. The code and models will be publicly available at https: //github.com/microsoft/SimMIM .

<div align="center">
<img src="https://user-images.githubusercontent.com/30762564/159404597-ac6d3a44-ee59-4cdc-8f6f-506a7d1b18b6.png" width="40%"/>
</div>

## Models and Benchmarks

Here, we report the results of the model, and more results will be coming soon.

| Backbone  | Pre-train epoch | Fine-tuning Top-1 |                                                         Pre-train Config                                                          |                                                              Fine-tuning Config                                                              |                                                                                                                           Download                                                                                                                            |
| :-------: | :-------------: | :---------------: | :-------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Swin-Base |       100       |       82.9        | [config](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simmim/simmim_swin-base_16xb128-coslr-100e_in1k-192) | [config](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/imagenet/swin-base_ft-8xb256-coslr-100e_in1k) | [model](https://download.openmmlab.com/mmselfsup/simmim/simmim_swin-base_16xb128-coslr-100e_in1k-192_20220316-1d090125.pth) \| [log](https://download.openmmlab.com/mmselfsup/simmim/simmim_swin-base_16xb128-coslr-100e_in1k-192_20220316-1d090125.log.json) |

## Citation

```bibtex
@inproceedings{xie2021simmim,
  title={SimMIM: A Simple Framework for Masked Image Modeling},
  author={Xie, Zhenda and Zhang, Zheng and Cao, Yue and Lin, Yutong and Bao, Jianmin and Yao, Zhuliang and Dai, Qi and Hu, Han},
  booktitle={International Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2022}
}
```
