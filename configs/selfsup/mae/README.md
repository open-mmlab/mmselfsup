# MAE Experimental

## Abstract

<!-- [ABSTRACT] -->

This paper shows that masked autoencoders (MAE) are
scalable self-supervised learners for computer vision. Our
MAE approach is simple: we mask random patches of the
input image and reconstruct the missing pixels. It is based
on two core designs. First, we develop an asymmetric
encoder-decoder architecture, with an encoder that operates only on the
visible subset of patches (without mask tokens), along with a lightweight
decoder that reconstructs the original image from the latent representation
and mask tokens. Second, we find that masking a high proportion
of the input image, e.g., 75%, yields a nontrivial and
meaningful self-supervisory task. Coupling these two designs enables us to
train large models efficiently and effectively: we accelerate
training (by 3× or more) and improve accuracy. Our scalable approach allows
for learning high-capacity models that generalize well: e.g., a vanilla
ViT-Huge model achieves the best accuracy (87.8%) among
methods that use only ImageNet-1K data. Transfer performance in downstream tasks outperforms supervised pretraining and shows promising scaling behavior.

<!-- [IMAGE] -->
<div align="center">
<img src="../../../resources/model_zoo/mae.png" width="40%"/>
</div>


## Models and Benchmarks

Here, we report the results of the model, which is pre-trained on ImageNet1K
for 400 epochs, the details are below:



| Backbone | Pre-train epoch | Fine-tuning Top-1 |                  Pre-train Config                  |                                       Fine-tuning Config                                       |                                                                                                                            Download                                                                                                                            |
| :------: | :-------------: | :---------------: | :------------------------------------------------: | :--------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ViT-B/16 |       400       |       82.93       | [config](./mae_vit-b-16_8xb512-coslr-400e_in1k.py) | [config](../../benchmarks/classification/imagenet/vit-b-16_8xb128-coslr-100e-finetune_in1k.py) | [model](https://download.openmmlab.com/mmselfsup/mae/mae_vit-b-16_8xb512-coslr-400e_in1k_20220107-4f3b0ae2.pth) &#124; [log](https://github.com/open-mmlab/mmselfsup/blob/feature/mae_experimental/configs/selfsup/mae/mae_vit-b-16_8xb512-coslr-400e_in1k.py) |


## Citation

<!-- [ALGORITHM] -->

```bibtex
@article{He2021MaskedAA,
  title={Masked Autoencoders Are Scalable Vision Learners},
  author={Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and
  Piotr Doll'ar and Ross B. Girshick},
  journal={ArXiv},
  year={2021},
  volume={abs/2111.06377}
}
```