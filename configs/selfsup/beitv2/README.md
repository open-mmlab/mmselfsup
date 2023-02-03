# BEiT v2

> [BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/abs/2208.06366)

<!-- [ALGORITHM] -->

## Abstract

Masked image modeling (MIM) has demonstrated impressive results in self-supervised representation learning by recovering corrupted image patches. However, most existing studies operate on low-level image pixels, which hinders the exploitation of high-level semantics for representation models. In this work, we propose to use a semantic-rich visual tokenizer as the reconstruction target for masked prediction, providing a systematic way to promote MIM from pixel-level to semantic-level. Specifically, we propose vector-quantized knowledge distillation to train the tokenizer, which discretizes a continuous semantic space to compact codes. We then pretrain vision Transformers by predicting the original visual tokens for the masked image patches. Furthermore, we introduce a patch aggregation strategy which associates discrete image patches to enhance global semantic representation. Experiments on image classification and semantic segmentation show that BEiT v2 outperforms all compared MIM methods. On ImageNet-1K (224 size), the base-size BEiT v2 achieves 85.5% top-1 accuracy for fine-tuning and 80.1% top-1 accuracy for linear probing. The large-size BEiT v2 obtains 87.3% top-1 accuracy for ImageNet-1K (224 size) fine-tuning, and 56.7% mIoU on ADE20K for semantic segmentation.

<div align="center">
<img src="https://user-images.githubusercontent.com/36138628/203912182-5967a520-d455-49ea-bc67-dcbd500d76bf.png" width="70%"/>
</div>

## Models and Benchmarks

During trainig, the `VQKD` target generator will download **VQ-KD** model automatically. Besides, You could also download **VQ-KD** model from this [link](https://download.openmmlab.com/mmselfsup/1.x/target_generator_ckpt/vqkd_encoder.pth) manually.

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
      <th>Linear Eval</th>
      <th>Fine-tuning</th>
      <th>Pretrain</th>
      <th>Linear Eval</th>
      <th>Fine-tuning</th>
	</tr>
  </thead>
  <tr>
      <td>BEiT v2</td>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>2048</td>
      <td>/</td>
      <td>85.0</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/beitv2/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/beitv2/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221212-a157be30.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/beitv2/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221206_012130.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/beitv2/classification/vit-base-p16_ft-8xb128-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/beitv2/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221212-d1c0789e.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/beitv2/beitv2_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221211_155017.json'>log</a></td>
	</tr>
  </tbody>
</table>

## Citation

```bibtex
@article{beitv2,
    title={{BEiT v2}: Masked Image Modeling with Vector-Quantized Visual Tokenizers},
    author={Zhiliang Peng and Li Dong and Hangbo Bao and Qixiang Ye and Furu Wei},
    journal={ArXiv},
    year={2022}
}
```
