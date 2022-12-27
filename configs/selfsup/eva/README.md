# EVA

> [EVA: Exploring the Limits of Masked Visual Representation Learning at Scale
> ](https://arxiv.org/abs/2211.07636)

<!-- [ALGORITHM] -->

## Abstract

We launch EVA, a vision-centric foundation model to explore the limits of visual representation at scale using only publicly accessible data. EVA is a vanilla ViT pre-trained to reconstruct the masked out image-text aligned vision features conditioned on visible image patches. Via this pretext task, we can efficiently scale up EVA to one billion parameters, and sets new records on a broad range of representative vision downstream tasks, such as image recognition, video action recognition, object detection, instance segmentation and semantic segmentation without heavy supervised training. Moreover, we observe quantitative changes in scaling EVA result in qualitative changes in transfer learning performance that are not present in other models. For instance, EVA takes a great leap in the challenging large vocabulary instance segmentation task: our model achieves almost the same state-of-the-art performance on LVISv1.0 dataset with over a thousand categories and COCO dataset with only eighty categories. Beyond a pure vision encoder, EVA can also serve as a vision-centric, multi-modal pivot to connect images and text. We find initializing the vision tower of a giant CLIP from EVA can greatly stabilize the training and outperform the training from scratch counterpart with much fewer samples and less compute, providing a new direction for scaling up and accelerating the costly training of multi-modal foundation models. To facilitate future research, we release all the code and models at this https URL.

<div align="center">
<img src="https://user-images.githubusercontent.com/56866854/207797596-ada38a11-7fee-4f06-b764-02bcdd2ab569.png" width="80%"/>
</div>

## Models and Benchmarks

Here, we report the results of the model, which is pre-trained on ImageNet-1k
for 400 epochs, the details are below:

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
      <th>Linear Eval</th>
      <th>Fine-tuning</th>
      <th>Pretrain</th>
      <th>Linear Eval</th>
      <th>Fine-tuning</th>
	</tr>
  </thead>
  <tbody>
  <tr>
      <td rowspan="1">EVA</td>
	    <td>ViT-B/16</td>
	    <td>400</td>
      <td>4096</td>
      <td>69.02</td>
      <td>83.72</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k_20221226-26d90f07.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k_20221220_113809.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/elfsup/eva/classification/vit-base-p16_linear-8xb2048-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k_20221226-ef51bf09.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k_20221222_134137.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/eva/classification/vit-base-p16_ft-8xb128-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221226-f61cf992.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221221_212618.json'>log</a></td>
	</tr>
</tbody>
</table>

## Citation

```bibtex
@article{fang2022eva,
  title={Eva: Exploring the limits of masked visual representation learning at scale},
  author={Fang, Yuxin and Wang, Wen and Xie, Binhui and Sun, Quan and Wu, Ledell and Wang, Xinggang and Huang, Tiejun and Wang, Xinlong and Cao, Yue},
  journal={arXiv preprint arXiv:2211.07636},
  year={2022}
}
```
