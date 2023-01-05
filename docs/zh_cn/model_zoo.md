# 模型库

本部分内容主要介绍 MMSelfSup 支持的模型和部分下游任务的评测结果。

- [模型库](#模型库)
  - [下游任务评测](#下游任务评测)
    - [ImageNet](#imagenet)

## 下游任务评测

### ImageNet

ImageNet 有多个版本，不过最常用的是 ILSVRC 2012。我们提供了基于各类算法的预训练模型的分类结果，包括线性评估和微调，同时有对应的模型和日志文件。

<table class="docutils">
<thead>
  <tr>
	    <th rowspan="2">算法</th>
	    <th rowspan="2">主干</th>
	    <th rowspan="2">预训练 Epoch</th>
      <th rowspan="2">Batch 大小</th>
      <th colspan="2" align="center">结果 (Top-1 %)</th>
      <th colspan="3" align="center">链接</th>
	</tr>
	<tr>
      <th>线性评估</th>
      <th>微调</th>
      <th>预训练</th>
      <th>线性评估</th>
      <th>微调</th>
	</tr>
  </thead>
  <tbody>
  <tr>
	    <td>Relative-Loc</td>
	    <td>ResNet50</td>
	    <td>70</td>
      <td>512</td>
      <td>40.4</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k/relative-loc_resnet50_8xb64-steplr-70e_in1k_20220825-daae1b41.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k/relative-loc_resnet50_8xb64-steplr-70e_in1k_20220802_223045.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-steplr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220825-c2a0b188.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220804_194226.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>Rotation-Pred</td>
	    <td>ResNet50</td>
	    <td>70</td>
      <td>128</td>
      <td>47.0</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k/rotation-pred_resnet50_8xb16-steplr-70e_in1k_20220825-a8bf5f69.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k/rotation-pred_resnet50_8xb16-steplr-70e_in1k_20220805_113136.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-steplr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220825-7c6edcb3.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220808_143921.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>NPID</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>58.3</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/npid/npid_resnet50_8xb32-steplr-200e_in1k/npid_resnet50_8xb32-steplr-200e_in1k_20220825-a67c5440.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/npid/npid_resnet50_8xb32-steplr-200e_in1k/npid_resnet50_8xb32-steplr-200e_in1k_20220725_161221.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-steplr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/npid/npid_resnet50_8xb32-steplr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220825-661b736e.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/npid/npid_resnet50_8xb32-steplr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220728_150535.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td rowspan="3">SimCLR</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>62.7</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_8xb32-coslr-200e_in1k/simclr_resnet50_8xb32-coslr-200e_in1k_20220825-15f807a4.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_8xb32-coslr-200e_in1k/simclr_resnet50_8xb32-coslr-200e_in1k_20220721_103223.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb512-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-9596a505.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220724_210050.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>4096</td>
      <td>66.9</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simclr/simclr_resnet50_16xb256-coslr-200e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-200e_in1k/simclr_resnet50_16xb256-coslr-200e_in1k_20220825-4d9cce50.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-200e_in1k/simclr_resnet50_16xb256-coslr-200e_in1k_20220721_150508.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb512-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-f12c0457.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220724_172050.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>800</td>
      <td>4096</td>
      <td>69.2</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simclr/simclr_resnet50_16xb256-coslr-800e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-800e_in1k/simclr_resnet50_16xb256-coslr-800e_in1k_20220825-85fcc4de.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-800e_in1k/simclr_resnet50_16xb256-coslr-800e_in1k_20220725_112248.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb512-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-800e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-b80ae1e5.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simclr/simclr_resnet50_16xb256-coslr-800e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220730_165101.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>MoCo v2</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>67.5</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/mocov2_resnet50_8xb32-coslr-200e_in1k_20220825-b6d23c86.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/mocov2_resnet50_8xb32-coslr-200e_in1k_20220721_215805.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-steplr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220825-994c4128.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220724_172046.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>BYOL</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>4096</td>
      <td>71.8</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/byol/byol_resnet50_16xb256-coslr-200e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220721_150515.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb512-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-7596c6f5.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220724_130251.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>SwAV</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>70.5</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96_20220825-5b3fc7fc.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96_20220728_141003.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96/resnet50_linear-8xb32-coslr-100e_in1k/resnet50_linear-8xb32-coslr-100e_in1k_20220825-80341e08.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96/resnet50_linear-8xb32-coslr-100e_in1k/resnet50_linear-8xb32-coslr-100e_in1k_20220802_145230.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>DenseCL</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>63.5</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/densecl_resnet50_8xb32-coslr-200e_in1k_20220825-3078723b.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/densecl_resnet50_8xb32-coslr-200e_in1k_20220727_221415.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-steplr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220825-f0f0a579.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/densecl/densecl_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb32-steplr-100e_in1k/resnet50_linear-8xb32-steplr-100e_in1k_20220730_091650.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td rowspan="2">SimSiam</td>
	    <td>ResNet50</td>
	    <td>100</td>
      <td>256</td>
      <td>68.3</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k/simsiam_resnet50_8xb32-coslr-100e_in1k_20220825-d07cb2e6.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k/simsiam_resnet50_8xb32-coslr-100e_in1k_20220725_224724.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb512-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-f53ba400.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220804_175115.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>69.8</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k/simsiam_resnet50_8xb32-coslr-200e_in1k_20220825-efe91299.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k/simsiam_resnet50_8xb32-coslr-200e_in1k_20220726_033722.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb512-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220825-519b5135.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k/resnet50_linear-8xb512-coslr-90e_in1k/resnet50_linear-8xb512-coslr-90e_in1k_20220802_120717.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
      <td>BarlowTwins</td>
	    <td>ResNet50</td>
	    <td>300</td>
      <td>2048</td>
      <td>71.8</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k/barlowtwins_resnet50_8xb256-coslr-300e_in1k_20220825-57307488.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k/barlowtwins_resnet50_8xb256-coslr-300e_in1k_20220726_033718.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k/resnet50_linear-8xb32-coslr-100e_in1k/resnet50_linear-8xb32-coslr-100e_in1k_20220825-52fde35f.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k/resnet50_linear-8xb32-coslr-100e_in1k/resnet50_linear-8xb32-coslr-100e_in1k_20220730_093018.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
      <td rowspan="6">MoCo v3</td>
	    <td>ResNet50</td>
	    <td>100</td>
      <td>4096</td>
      <td>69.6</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_20220927-f1144efa.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_20220915_154635.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb128-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220927-8f7d937e.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-100e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220920_113350.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>300</td>
      <td>4096</td>
      <td>72.8</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k/mocov3_resnet50_8xb512-amp-coslr-300e_in1k_20220927-1e4f3304.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k/mocov3_resnet50_8xb512-amp-coslr-300e_in1k_20220915_180538.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb128-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220927-d21ddac2.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-300e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220920_113403.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>800</td>
      <td>4096</td>
      <td>74.4</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/mocov3_resnet50_8xb512-amp-coslr-800e_in1k_20220927-e043f51a.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/mocov3_resnet50_8xb512-amp-coslr-800e_in1k_20220919_111209.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb128-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220927-0e97a483.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_resnet50_8xb512-amp-coslr-800e_in1k/resnet50_linear-8xb128-coslr-90e_in1k/resnet50_linear-8xb128-coslr-90e_in1k_20220926_102021.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ViT-small</td>
	    <td>300</td>
      <td>4096</td>
      <td>73.6</td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k-224_20220826-08bc52f7.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k-224_20220721_153833.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-small-p16_linear-8xb128-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k/vit-small-p16_linear-8xb128-coslr-90e_in1k/vit-small-p16_linear-8xb128-coslr-90e_in1k_20220826-376674ef.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-small-p16_16xb256-amp-coslr-300e_in1k/vit-small-p16_linear-8xb128-coslr-90e_in1k/vit-small-p16_linear-8xb128-coslr-90e_in1k_20220724_140850.json'>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>4096</td>
      <td>76.9</td>
      <td>83.0</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k-224_20220826-25213343.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k-224_20220725_104223.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_linear-8xb128-coslr-90e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/vit-base-p16_linear-8xb128-coslr-90e_in1k/vit-base-p16_linear-8xb128-coslr-90e_in1k_20220826-83be7758.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/vit-base-p16_linear-8xb128-coslr-90e_in1k/vit-base-p16_linear-8xb128-coslr-90e_in1k_20220729_004628.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_ft-8xb64-coslr-150e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb64-coslr-150e_in1k/vit-base-p16_ft-8xb64-coslr-150e_in1k_20220826-f1e6c442.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-base-p16_16xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb64-coslr-150e_in1k/vit-base-p16_ft-8xb64-coslr-150e_in1k_20220809_103500.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-large</td>
	    <td>300</td>
      <td>4096</td>
      <td>/</td>
      <td>83.7</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mocov3/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k-224_20220829-9b88a442.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k-224_20220818_143032.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-large-p16_ft-8xb64-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k/vit-large-p16_ft-8xb64-coslr-100e_in1k/vit-large-p16_ft-8xb64-coslr-100e_in1k_20220829-878a2f7f.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mocov3/mocov3_vit-large-p16_64xb64-amp-coslr-300e_in1k/vit-large-p16_ft-8xb64-coslr-100e_in1k/vit-large-p16_ft-8xb64-coslr-100e_in1k_20220825_201433.json'>log</a></td>
	</tr>
  <tr>
      <td rowspan="9">MAE</td>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>4096</td>
      <td>60.8</td>
      <td>83.1</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-base-p16_8xb512-amp-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220829-c2cf66ba.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k/mae_vit-base-p16_8xb512-coslr-300e-fp16_in1k_20220718_152424.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_linear-8xb2048-coslr-90e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k_20220720_104514.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_ft-8xb128-coslr-100e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220713_140138.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td>400</td>
      <td>4096</td>
      <td>62.5</td>
      <td>83.3</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-base-p16_8xb512-amp-coslr-400e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-base-p16_8xb512-coslr-400e-fp16_in1k_20220825-bc79e40b.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-base-p16_8xb512-coslr-400e-fp16_in1k_20220628_200815.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_linear-8xb2048-coslr-90e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-400e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k_20220713_142534.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_ft-8xb128-coslr-100e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220708_183134.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td>800</td>
      <td>4096</td>
      <td>65.1</td>
      <td>83.3</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-base-p16_8xb512-amp-coslr-800e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-base-p16_8xb512-coslr-800e-fp16_in1k_20220825-5d81fbc4.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-base-p16_8xb512-coslr-800e-fp16_in1k_20220718_134405.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_linear-8xb2048-coslr-90e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k20220721_203941.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_ft-8xb128-coslr-100e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-800e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220724_232940.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td>1600</td>
      <td>4096</td>
      <td>67.1</td>
      <td>83.5</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-base-p16_8xb512-amp-coslr-1600e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220825-f7569ca2.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k_20220815_103458.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_linear-8xb2048-coslr-90e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k/vit-base-p16_linear-8xb2048-coslr-90e_in1k_20220724_232557.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_ft-8xb128-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220825-cf70aa21.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-base-p16_8xb512-fp16-coslr-1600e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20220721_202304.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-large</td>
	    <td>400</td>
      <td>4096</td>
      <td>70.7</td>
      <td>85.2</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-large-p16_8xb512-amp-coslr-400e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k_20220825-b11d0425.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k_20220726_202204.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-large-p16_linear-8xb2048-coslr-90e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k/vit-large-p16_linear-8xb2048-coslr-90e_in1k/vit-large-p16_linear-8xb2048-coslr-90e_in1k_20220803_101331.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-large-p16_ft-8xb128-coslr-50e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-400e_in1k/vit-large-p16_ft-8xb128-coslr-50e_in1k/vit-large-p16_ft-8xb128-coslr-50e_in1k_20220729_122511.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-large</td>
	    <td>800</td>
      <td>4096</td>
      <td>73.7</td>
      <td>85.4</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-large-p16_8xb512-amp-coslr-800e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k_20220825-df72726a.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k_20220804_104018.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-large-p16_linear-8xb2048-coslr-90e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k/vit-large-p16_linear-8xb2048-coslr-90e_in1k/vit-large-p16_linear-8xb2048-coslr-90e_in1k_20220808_092730.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-large-p16_ft-8xb128-coslr-50e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-800e_in1k/vit-large-p16_ft-8xb128-coslr-50e_in1k/vit-large-p16_ft-8xb128-coslr-50e_in1k_20220730_235819.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-large</td>
	    <td>1600</td>
      <td>4096</td>
      <td>75.5</td>
      <td>85.7</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-large-p16_8xb512-amp-coslr-1600e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220825-cc7e98c9.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220806_210725.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-large-p16_linear-8xb2048-coslr-90e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/vit-large-p16_linear-8xb2048-coslr-90e_in1k/vit-large-p16_linear-8xb2048-coslr-90e_in1k_20220813_155615.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-large-p16_ft-8xb128-coslr-50e_in1k.py'>config</a> | model | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/vit-large-p16_ft-8xb128-coslr-50e_in1k/vit-large-p16_ft-8xb128-coslr-50e_in1k_20220813_125305.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-huge-FT-224</td>
	    <td>1600</td>
      <td>4096</td>
      <td>/</td>
      <td>86.9</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-huge-p16_8xb512-amp-coslr-1600e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220916-ff848775.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220814_135241.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-huge-p16_ft-8xb128-coslr-50e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/vit-huge-p16_ft-8xb128-coslr-50e_in1k/vit-huge-p16_ft-8xb128-coslr-50e_in1k_20220916-0bfc9bfd.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/vit-huge-p16_ft-8xb128-coslr-50e_in1k/vit-huge-p16_ft-8xb128-coslr-50e_in1k_20220829_114027.json'>log</a></td>
	</tr>
  <tr>
	    <td>ViT-huge-FT-448</td>
	    <td>1600</td>
      <td>4096</td>
      <td>/</td>
      <td>87.3</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mae/mae_vit-huge-p16_8xb512-amp-coslr-1600e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220916-ff848775.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k_20220814_135241.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-huge-p16_ft-32xb8-coslr-50e_in1k-448.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/vit-huge-p16_ft-32xb8-coslr-50e_in1k-448/vit-huge-p16_ft-32xb8-coslr-50e_in1k-448_20220916-95b6a0ce.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-huge-p16_8xb512-fp16-coslr-1600e_in1k/vit-huge-p16_ft-32xb8-coslr-50e_in1k-448/vit-huge-p16_ft-32xb8-coslr-50e_in1k-448_20220913_113737.json'>log</a></td>
	</tr>
  <tr>
      <td>CAE</td>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>2048</td>
      <td>/</td>
      <td>83.3</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/cae/cae_vit-base-p16_16xb128-amp-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/cae/cae_vit-base-p16_16xb128-fp16-coslr-300e_in1k/cae_vit-base-p16_16xb128-fp16-coslr-300e_in1k_20220825-404a1929.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/cae/cae_vit-base-p16_16xb128-fp16-coslr-300e_in1k/cae_vit-base-p16_16xb128-fp16-coslr-300e_in1k_20220615_163141.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_ft-8xb128-coslr-100e-rpe_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/cae/cae_vit-base-p16_16xb128-fp16-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e-rpe_in1k/vit-base-p16_ft-8xb128-coslr-100e-rpe_in1k_20220825-f3d234cd.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/cae/cae_vit-base-p16_16xb128-fp16-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e-rpe_in1k/vit-base-p16_ft-8xb128-coslr-100e-rpe_in1k_20220711_165500.json'>log</a></td>
	</tr>
  <tr>
      <td rowspan="4">SimMIM</td>
	    <td>Swin-base-FT192</td>
	    <td>100</td>
      <td>2048</td>
      <td>/</td>
      <td>82.7</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simmim/simmim_swin-base_16xb128-amp-coslr-100e_in1k-192.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192_20220829-0e15782d.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192_20220827_034052.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/swin-base_ft-8xb256-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/swin-base_ft-8xb256-coslr-100e_in1k/swin-base_ft-8xb256-coslr-100e_in1k_20220829-9cf23aa1.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/swin-base_ft-8xb256-coslr-100e_in1k/swin-base_ft-8xb256-coslr-100e_in1k_20220829_001452.json'>log</a></td>
	</tr>
  <tr>
	    <td>Swin-base-FT224</td>
	    <td>100</td>
      <td>2048</td>
      <td>/</td>
      <td>83.5</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simmim/simmim_swin-base_16xb128-amp-coslr-100e_in1k-192.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192_20220829-0e15782d.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192_20220827_034052.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/swin-base_ft-8xb256-coslr-100e_in1k-224.py'>config</a> | model | log</td>
	</tr>
  <tr>
	    <td>Swin-base-FT224</td>
	    <td>800</td>
      <td>2048</td>
      <td>/</td>
      <td>83.7</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simmim/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192_20220916-a0e931ac.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192_20220906_141645.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/swin-base_ft-8xb256-coslr-100e_in1k-224.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192/swin-base_ft-8xb256-coslr-100e_in1k-224/swin-base_ft-8xb256-coslr-100e_in1k-224_20221208-155cc6e6.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_16xb128-amp-coslr-800e_in1k-192/swin-base_ft-8xb256-coslr-100e_in1k-224/swin-base_ft-8xb256-coslr-100e_in1k-224_20221207_135847.json'>log</a></td>
	</tr>
  <tr>
	    <td>Swin-large-FT224</td>
	    <td>800</td>
      <td>2048</td>
      <td>/</td>
      <td>84.8</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192_20220916-4ad216d3.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192_20220907_203738.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/swin-large_ft-8xb256-coslr-ws14-100e_in1k-224.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/swin-large_ft-8xb256-coslr-ws14-100e_in1k-224/swin-large_ft-8xb256-coslr-ws14-100e_in1k-224_20220916-d4865790.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-large_16xb128-amp-coslr-800e_in1k-192/swin-large_ft-8xb256-coslr-ws14-100e_in1k-224/swin-large_ft-8xb256-coslr-ws14-100e_in1k-224_20220914_133331.json'>log</a></td>
	</tr>
  <tr>
      <td>MaskFeat</td>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>2048</td>
      <td>/</td>
      <td>83.4</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221101-6dfc8bf3.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221019_194256.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_ft-8xb256-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb256-coslr-100e_in1k/vit-base-p16_ft-8xb256-coslr-100e_in1k_20221028-5134431c.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/maskfeat/maskfeat_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb256-coslr-100e_in1k/vit-base-p16_ft-8xb256-coslr-100e_in1k_20221026_105344.json'>log</a></td>
	</tr>
  <tr>
      <td>BEiT</td>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>2048</td>
      <td>/</td>
      <td>83.1</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/beit/beit_vit-base-p16_8xb256-amp-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/beit/beit_vit-base-p16_8xb256-amp-coslr-300e_in1k/beit_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221128-ab79e626.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/beit/beit_vit-base-p16_8xb256-amp-coslr-300e_in1k/beit_vit-base-p16_8xb256-amp-coslr-300e_in1k_20221123_103802.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/beit/classification/vit-base-p16_ft-8xb128-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/beit/beit_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221128-0ca393e9.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/beit/beit_vit-base-p16_8xb256-amp-coslr-300e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221127_162126.json'>log</a></td>
	</tr>
  <tr>
  <td>MILAN</td>
	    <td>ViT-base</td>
	    <td>400</td>
      <td>4096</td>
      <td>78.9</td>
      <td>85.3</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k_20221129-180922e8.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k_20221123_112721.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/milan/classification/vit-base-p16_linear-8xb2048-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k-milan_20221129-74ac94fa.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k-milan_20221125_031826.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/milan/classification/vit-base-p16_linear-8xb2048-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k_20221129-03f26f85.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/milan/milan_vit-base-p16_16xb256-amp-coslr-400e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k_20221124_215401.json'>log</a></td>
  </tr>
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
  <tr>
      <td>EVA</td>
	    <td>ViT-base</td>
	    <td>400</td>
      <td>4096</td>
      <td>69.0</td>
      <td>83.7</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k_20221226-26d90f07.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k_20221220_113809.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/elfsup/eva/classification/vit-base-p16_linear-8xb2048-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k_20221226-ef51bf09.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k/vit-base-p16_linear-8xb2048-coslr-100e_in1k_20221222_134137.json'>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/eva/classification/vit-base-p16_ft-8xb128-coslr-100e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221226-f61cf992.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/eva/eva-mae-style_vit-base-p16_16xb256-coslr-400e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k/vit-base-p16_ft-8xb128-coslr-100e_in1k_20221221_212618.json'>log</a></td>
  </tr>
<tr>
      <td>MixMIM</td>
	    <td>MixMIM-Base</td>
      <td>400</td>
      <td>2048</td>
      <td>/</td>
      <td>84.6</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k/mixmim-base-p16_16xb128-coslr-300e_in1k_20221208-44fe8d2c.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k/mixmim-base-p16_16xb128-coslr-300e_in1k_20221204_134711.json'>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/selfsup/mixmim/classification/mixmim-base-p16_ft-8xb128-coslr-100e-in1k.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k/mixmim-base-p16_ft-8xb128-coslr-100e_in1k/mixmim-base-p16_ft-8xb128-coslr-100e_in1k_20221208-41ecada9.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/mixmim/mixmim-base-p16_16xb128-coslr-300e_in1k/mixmim-base-p16_ft-8xb128-coslr-100e_in1k/mixmim-base-p16_ft-8xb128-coslr-100e_in1k_20221206_143046.json'>log</a></td>
</tr>
</tbody>
</table>
