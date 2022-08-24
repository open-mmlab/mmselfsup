# Model Zoo

All models and part of benchmark results are recorded below.

- [Model Zoo](#model-zoo)
  - [Statistics](#statistics)
  - [Benchmark](#benchmark)
    - [ImageNet](#imagenet)

## Statistics

- Number of papers: 17

- Number of checkpoints: xx ckpts

## Benchmark

ImageNet has multiple versions, but the most commonly used one is ILSVRC 2012. The results below are trained by linear evaluation or fine-tuning and the backbone is loaded with self-supervised pretrain backbone.

### ImageNet

<table>
	<tr>
	    <th>Algorithm</th>
	    <th>Backbone</th>
	    <th>Epoch</th>
      <th>Batch Size</th>
      <th>Linear Eval</th>
      <th>Fine-tuning</th>
      <th>Pretrain Link</th>
      <th>Classification Link</th>
	</tr>
  <tr>
	    <td>Relative-Loc</td>
	    <td>ResNet50</td>
	    <td>70</td>
      <td>512</td>
      <td>40.4</td>
      <td></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
      <td><a href=''>config</a> | <a href=''>model</a> | <a href=''>log</a></td>
	</tr>
  <tr>
	    <td>Rotation-Pred</td>
	    <td>ResNet50</td>
	    <td>70</td>
      <td>128</td>
      <td>47.0</td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>NPID</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>58.3</td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td rowspan="3">SimCLR</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>62.7</td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>4096</td>
      <td>66.9</td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>800</td>
      <td>4096</td>
      <td>69.2</td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>MoCo v2</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>62.7</td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>BYOL</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>4096</td>
      <td>71.8</td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>SwAV</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>70.5</td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>DenseCL</td>
	    <td>ResNet50</td>
	    <td>200</td>
      <td>256</td>
      <td>63.5</td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td rowspan="2">SimSiam</td>
	    <td>ResNet50</td>
	    <td>100</td>
      <td>256</td>
      <td>68.3</td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>100</td>
      <td>256</td>
      <td>69.8</td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
      <td>BarlowTwins</td>
	    <td>ResNet50</td>
	    <td>300</td>
      <td>2048</td>
      <td>71.8</td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
      <td rowspan="6">MoCo v3</td>
	    <td>ResNet50</td>
	    <td>100</td>
      <td>4096</td>
      <td>69.4</td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>ResNet50</td>
	    <td>300</td>
      <td>4096</td>
      <td>73.1</td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>ViT-small</td>
	    <td>300</td>
      <td>4096</td>
      <td>73.6</td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>4096</td>
      <td>76.9</td>
      <td>83.0</td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td>800</td>
      <td>4096</td>
      <td></td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>ViT-large</td>
	    <td>300</td>
      <td>4096</td>
      <td></td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
      <td rowspan="6">MAE</td>
	    <td>ViT-base</td>
	    <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
	    <td>ViT-base</td>
	    <td></td>
      <td></td>
      <td></td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
      <td>CAE</td>
	    <td>ViT-base</td>
	    <td>300</td>
      <td>2048</td>
      <td></td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>
  <tr>
      <td>SimMIM</td>
	    <td>Swin-base</td>
	    <td>300</td>
      <td>2048</td>
      <td></td>
      <td></td>
      <td>TODO</td>
      <td>TODO</td>
	</tr>