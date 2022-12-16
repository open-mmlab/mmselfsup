# MixMIM

> [MixMIM: Mixed and Masked Image Modeling for Efficient Visual Representation Learning](https://arxiv.org/abs/2205.13137)

<!-- [ALGORITHM] -->

## Abstract

In this study, we propose Mixed and Masked Image Modeling (MixMIM), a
simple but efficient MIM method that is applicable to various hierarchical Vision
Transformers. Existing MIM methods replace a random subset of input tokens with
a special \[MASK\] symbol and aim at reconstructing original image tokens from
the corrupted image. However, we find that using the \[MASK\] symbol greatly
slows down the training and causes training-finetuning inconsistency, due to the
large masking ratio (e.g., 40% in BEiT). In contrast, we replace the masked tokens
of one image with visible tokens of another image, i.e., creating a mixed image.
We then conduct dual reconstruction to reconstruct the original two images from
the mixed input, which significantly improves efficiency. While MixMIM can
be applied to various architectures, this paper explores a simpler but stronger
hierarchical Transformer, and scales with MixMIM-B, -L, and -H. Empirical
results demonstrate that MixMIM can learn high-quality visual representations
efficiently. Notably, MixMIM-B with 88M parameters achieves 85.1% top-1
accuracy on ImageNet-1K by pretraining for 600 epochs, setting a new record for
neural networks with comparable model sizes (e.g., ViT-B) among MIM methods.
Besides, its transferring performances on the other 6 datasets show MixMIM has
better FLOPs / performance tradeoff than previous MIM methods

<div align=center>
<img src="https://user-images.githubusercontent.com/56866854/202853730-d26fb3d7-e5e8-487a-aad5-e3d4600cef87.png"/>
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
      <th colspan="1" align="center">Results (Top-1 %)</th>
      <th colspan="2" align="center">Links</th>
	</tr>
	<tr>
      <th>Fine-tuning</th>
      <th>Pretrain</th>
      <th>Fine-tuning</th>
	</tr>
  </thead>
  <tr>
      <td>MixMIM</td>
	    <td>MixMIM-base</td>
	    <td>300</td>
      <td>2048</td>
      <td>84.63</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k/mixmim-base-p16_16xb128-coslr-300e_in1k_20221208-44fe8d2c.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k/mixmim-base-p16_16xb128-coslr-300e_in1k_20221204_134711.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mixmim/classification/mixmim-base-p16_ft-8xb128-coslr-100e-in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k/mixmim-base-p16_ft-8xb128-coslr-100e_in1k/mixmim-base-p16_ft-8xb128-coslr-100e_in1k_20221208-41ecada9.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k/mixmim-base-p16_ft-8xb128-coslr-100e_in1k/mixmim-base-p16_ft-8xb128-coslr-100e_in1k_20221206_143046.json'>log</a></td>
	</tr>
  </tbody>
</table>

## Citation

```bibtex
@article{MixMIM2022,
  author  = {Jihao Liu, Xin Huang, Yu Liu, Hongsheng Li},
  journal = {arXiv:2205.13137},
  title   = {MixMIM: Mixed and Masked Image Modeling for Efficient Visual Representation Learning},
  year    = {2022},
}
```
