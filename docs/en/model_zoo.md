# Model Zoo

All models and part of benchmark results are recorded below.

- [Model Zoo](#model-zoo)
  - [Statistics](#statistics)
  - [Benchmarks](#benchmarks)
    - [ImageNet](#imagenet)

## Statistics

- Number of papers: 17

- Number of checkpoints: xx ckpts

## Benchmarks

### ImageNet

ImageNet has multiple versions, but the most commonly used one is ILSVRC 2012. The classification results (Top-1 %) below are trained by linear evaluation or fine-tuning and the backbone is loaded with self-supervised pretrain backbone.

<table class="docutils" style='width=1500'>
<thead>
	<tr>
	    <th>Algorithm</th>
	    <th>Backbone</th>
	    <th>Epoch</th>
      <th>Batch Size</th>
      <th>Linear Eval</th>
      <th>Fine-tuning</th>
      <th>Pretrain Link</th>
      <th>Linear Eval Link</th>
      <th>Fine-tuning Link</th>
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
      <td>62.7</td>
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
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-coslr-100e_in1k.py'>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>DenseCL</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>63.5</td>
      <td>/</td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-steplr-100e_in1k.py'>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td rowspan="2">SimSiam</td>
	    <td>ResNet50</td>
	    <td>100</td>
      <td>256</td>
      <td>68.3</td>
      <td>/</td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb512-coslr-90e_in1k.py'>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>69.8</td>
      <td>/</td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb512-coslr-90e_in1k.py'>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
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
      <td rowspan="4">MoCo v3</td>
	    <td>ResNet50</td>
	    <td>100</td>
      <td>4096</td>
      <td>69.4</td>
      <td>/</td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb128-coslr-90e_in1k.py'>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>300</td>
      <td>4096</td>
      <td>73.1</td>
      <td>/</td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/resnet50_linear-8xb128-coslr-90e_in1k.py'>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td>/</td>
	</tr>
  <tr>
	    <td>ViT-small</td>
	    <td>300</td>
      <td>4096</td>
      <td>73.6</td>
      <td></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-small-p16_linear-8xb128-coslr-90e_in1k.py'>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>4096</td>
      <td>76.9</td>
      <td>83.0</td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_linear-8xb128-coslr-90e_in1k.py'>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
	</tr>
  <tr>
      <td rowspan="7">MAE</td>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>4096</td>
      <td>60.8</td>
      <td>83.1</td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td>400</td>
      <td>4096</td>
      <td>62.5</td>
      <td>83.3</td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td>800</td>
      <td>4096</td>
      <td>65.1</td>
      <td>83.3</td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td>1600</td>
      <td>4096</td>
      <td>67.1</td>
      <td>83.5</td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
	</tr>
  <tr>
	    <td>ViT-large</td>
	    <td>400</td>
      <td>4096</td>
      <td>70.7</td>
      <td>85.2</td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
	</tr>
  <tr>
	    <td>ViT-large</td>
	    <td>800</td>
      <td>4096</td>
      <td>73.7</td>
      <td>85.4</td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
	</tr>
  <tr>
	    <td>ViT-large</td>
	    <td>1600</td>
      <td>4096</td>
      <td>75.5</td>
      <td>85.7</td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
	</tr>
  <tr>
      <td>CAE</td>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>2048</td>
      <td>/</td>
      <td>83.3</td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/vit-base-p16_ft-8xb128-coslr-100e-rpe_in1k.py'>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
	</tr>
  <tr>
      <td>SimMIM</td>
	    <td>Swin-base</td>
	    <td>300</td>
      <td>2048</td>
      <td>/</td>
      <td>82.9</td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td>/</td>
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/swin-base_ft-8xb256-coslr-100e_in1k.py'>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
	</tr>
</tbody>
</table>
