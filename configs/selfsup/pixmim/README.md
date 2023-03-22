# PixMIM

> [PixMIM: Rethinking Pixel Reconstruction in Masked Image Modeling
> ](https://arxiv.org/abs/2303.02416)

## TL;DR

PixMIM can seamlessly replace MAE as a stronger baseline, with
negligible computational overhead.

<!-- [ALGORITHM] -->

## Abstract

Masked Image Modeling (MIM) has achieved promising progress with the advent of Masked Autoencoders
(MAE) and BEiT. However, subsequent works have complicated the framework with new auxiliary tasks or extra pretrained models,
inevitably increasing computational overhead. This paper undertakes a fundamental analysis of
MIM from the perspective of pixel reconstruction, which
examines the input image patches and reconstruction target, and highlights two critical but previously overlooked
bottlenecks. Based on this analysis, we propose a remarkably simple and effective method, PixMIM, that entails two
strategies: 1) filtering the high-frequency components from
the reconstruction target to de-emphasize the network’s focus on texture-rich details and 2) adopting a conservative
data transform strategy to alleviate the problem of missing foreground in MIM training. PixMIM can be easily
integrated into most existing pixel-based MIM approaches
(i.e., using raw images as reconstruction target) with negligible additional computation. Without bells and whistles,
our method consistently improves three MIM approaches,
MAE, ConvMAE, and LSMAE, across various downstream
tasks. We believe this effective plug-and-play method will
serve as a strong baseline for self-supervised learning and
provide insights for future improvements of the MIM framework.

<div align=center>
<img src="https://user-images.githubusercontent.com/30762564/226782993-28b2b20f-9143-4514-8c61-1aa81146d159.png"/>
</div>

## Models and Benchmarks

Here, we report the results of the model on ImageNet, the details are below:

<table class="docutils">
<thead>
  <tr>
	    <th rowspan="2">Algorithm</th>
	    <th rowspan="2">Backbone</th>
	    <th rowspan="2">Epoch</th>
      <th rowspan="2">Batch Size</th>
      <th colspan="2" align="center">Results (Top-1 %)</th>
      <th colspan="3" align="center">Links</th>
	</tr>
	<tr>
      <th>Linear Probing</th>
      <th>Fine-tuning</th>
      <th>Pretrain</th>
      <th>Linear Probing</th>
      <th>Fine-tuning</th>
	</tr>
  </thead>
    <tr>
      <td>PixMIM</td>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>4096</td>
      <td>63.3</td>
      <td>83.1</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-300e_in1k.py'> config </a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-300e_in1k/pixmim_vit-base-p16_8xb512-amp-coslr-300e_in1k_20230322-3304a88c.pth'> model </a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-300e_in1k/pixmim_vit-base-p16_8xb512-amp-coslr-300e_in1k_20230322-3304a88c.json'> log </a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/pixmim/classification/vit-base-p16_linear-8xb2048-coslr-torchvision-transform-90e_in1k.py'> config </a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-300e_in1k/vit-base-p16_linear-8xb2048-torchvision-transform-coslr-90e_in1k/vit-base-p16_linear-8xb2048-torchvision-transform-coslr-90e_in1k_20230322-72322af8.pth'> model </a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-300e_in1k/vit-base-p16_linear-8xb2048-torchvision-transform-coslr-90e_in1k/vit-base-p16_linear-8xb2048-torchvision-transform-coslr-90e_in1k_20230322-72322af8.json'> log </a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/pixmim/classification/vit-base-p16_ft-8xb128-coslr-100e_in1k.py'> config </a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20230322-7eba2bc2.pth'> model </a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20230322-7eba2bc2.json'> log </a></td>
	</tr>
    <tr>
      <td>PixMIM</td>
	    <td>ViT-base</td>
	    <td>800</td>
      <td>4096</td>
      <td>67.5</td>
      <td>83.5</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-800e_in1k.py'> config </a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-800e_in1k/pixmim_vit-base-p16_8xb512-amp-coslr-800e_in1k_20230322-e8137924.pth'> model </a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-800e_in1k/pixmim_vit-base-p16_8xb512-amp-coslr-800e_in1k_20230322-e8137924.json'> log </a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/pixmim/classification/vit-base-p16_linear-8xb2048-coslr-torchvision-transform-90e_in1k.py'> config </a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-800e_in1k/vit-base-p16_linear-8xb2048-torchvision-transform-coslr-90e_in1k/vit-base-p16_linear-8xb2048-torchvision-transform-coslr-90e_in1k_20230322-12c15568.pth'> model </a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-800e_in1k/vit-base-p16_linear-8xb2048-torchvision-transform-coslr-90e_in1k/vit-base-p16_linear-8xb2048-torchvision-transform-coslr-90e_in1k_20230322-12c15568.json'> log </a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/1.x/configs/selfsup/pixmim/classification/vit-base-p16_ft-8xb128-coslr-100e_in1k.py'> config </a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-800e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20230322-616b1a7f.pth'> model </a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-800e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20230322-616b1a7f.json'> log </a></td>
	</tr>
  </tbody>
</table>

## Pre-train and Evaluation

### Pre-train

If you use a cluster managed by Slurm

```sh
# all of our experiments can be run on a single machine, with 8 A100 GPUs
bash tools/slurm_train.sh $partition $job_name configs/selfsup/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-300e_in1k.py --amp
```

If you use a single machine without any cluster management software

```sh
bash tools/dist_train.sh configs/selfsup/pixmim/pixmim_vit-base-p16_8xb512-amp-coslr-300e_in1k.py 8 --amp
```

### Linear Probing

If you use a cluster managed by Slurm

```sh
# all of our experiments can be run on a single machine, with 8 A100 GPUs
bash tools/benchmarks/classification/mim_slurm_train.sh $partition configs/selfsup/pixmim/classification/vit-base-p16_linear-8xb2048-coslr-torchvision-transform-90e_in1k.py --amp
```

If you use a single machine without any cluster management software

```sh
bash tools/benchmarks/classification/mim_dist_train.sh configs/selfsup/pixmim/classification/vit-base-p16_linear-8xb2048-coslr-torchvision-transform-90e_in1k.py 8 --amp
```

### Fine-tuning

If you use a cluster managed by Slurm

```sh
# all of our experiments can be run on a single machine, with 8 A100 GPUs
bash tools/benchmarks/classification/mim_slurm_train.sh $partition configs/selfsup/pixmim/classification/vit-base-p16_ft-8xb128-coslr-100e_in1k.py $pretrained_model --amp
```

If you use a single machine without any cluster management software

```sh
GPUS=8 bash tools/benchmarks/classification/mim_dist_train.sh configs/selfsup/pixmim/classification/vit-base-p16_ft-8xb128-coslr-100e_in1k.py $pretrained_model --amp
```

## Detection and Segmentation

If you want to evaluate your model on detection or segmentation task, we provide a [script](https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/tools/model_converters/mmcls2timm.py) to convert the model keys from MMClassification style to timm style.

```sh
cd $MMSELFSUP
python tools/model_converters/mmcls2timm.py $src_ckpt $dst_ckpt
```

Then, using this converted ckpt, you can evaluate your model on detection task, following [Detectron2](https://github.com/facebookresearch/detectron2/tree/main/projects/ViTDet)，
and on semantic segmentation task, following this [project](https://github.com/implus/mae_segmentation). Besides, using the unconverted ckpt, you can use
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/mae) to evaluate your model.

## Citation

```bibtex
@article{PixMIM,
  author  = {Yuan Liu, Songyang Zhang, Jiacheng Chen, Kai Chen, Dahua Lin},
  journal = {arXiv:2303.02416},
  title   = {PixMIM: Rethinking Pixel Reconstruction in Masked Image Modeling},
  year    = {2023},
}
```
