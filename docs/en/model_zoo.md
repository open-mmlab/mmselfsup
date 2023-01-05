# Model Zoo

All models and part of benchmark results are recorded below.

<<<<<<< HEAD
- [Model Zoo](#model-zoo)
  - [Benchmarks](#benchmarks)
    - [ImageNet](#imagenet)
=======
## Pre-trained models

| Algorithm                                                                                                          | Config                                                                                                                                                                                       | Download                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Relative Location](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/README.md)    | [relative-loc_resnet50_8xb64-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)               | [model](https://download.openmmlab.com/mmselfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k_20220225-84784688.pth) \| [log](https://download.openmmlab.com/mmselfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k_20220211_124808.log.json)     |
| [Rotation Prediction](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/README.md) | [rotation-pred_resnet50_8xb16-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py)            | [model](https://download.openmmlab.com/mmselfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k_20220225-5b9f06a0.pth) \| [log](https://download.openmmlab.com/mmselfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k_20220215_185303.log.json) |
| [DeepCluster](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/deepcluster/README.md)           | [deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/deepcluster/deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py)    | [model](https://download.openmmlab.com/mmselfsup/deepcluster/deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k-bb8681e2.pth)                                                                                                                                              |
| [NPID](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/npid/README.md)                         | [npid_resnet50_8xb32-steplr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                                     | [model](https://download.openmmlab.com/mmselfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k_20220225-5fbbda2a.pth) \| [log](https://download.openmmlab.com/mmselfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k_20220215_185513.log.json)                                   |
| [ODC](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/odc/README.md)                           | [odc_resnet50_8xb64-steplr-440e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py)                                        | [model](https://download.openmmlab.com/mmselfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k_20220225-a755d9c0.pth)   \| [log](https://download.openmmlab.com/mmselfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k_20220215_235245.log.json)                                     |
| [SimCLR](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simclr/README.md)                     | [simclr_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)                                 | [model](https://download.openmmlab.com/mmselfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_20220428-46ef6bb9.pth) \| [log](https://download.openmmlab.com/mmselfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k_20220411_182427.log.json)                             |
|                                                                                                                    | [simclr_resnet50_16xb256-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simclr/simclr_resnet50_16xb256-coslr-200e_in1k.py)                             | [model](https://download.openmmlab.com/mmselfsup/simclr/simclr_resnet50_16xb256-coslr-200e_in1k_20220428-8c24b063.pth) \| [log](https://download.openmmlab.com/mmselfsup/simclr/simclr_resnet50_16xb256-coslr-200e_in1k_20220423_205520.log.json)                         |
| [MoCo v2](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov2/README.md)                    | [mocov2_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                                 | [model](https://download.openmmlab.com/mmselfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k_20220225-89e03af4.pth) \| [log](https://download.openmmlab.com/mmselfsup/moco/mocov2_resnet50_8xb32-coslr-200e_in1k_20220210_110905.log.json)                                 |
| [BYOL](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/README.md)                         | [byol_resnet50_8xb32-accum16-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)                       | [model](https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k_20220225-5c8b2c2e.pth) \| [log](https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k_20220214_115709.log.json)                     |
|                                                                                                                    | [byol_resnet50_16xb256-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_16xb256-coslr-200e_in1k.py)                                   | [model](https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_16xb256-coslr-200e_in1k_20220527-b6f8eedd.pth) \| [log](https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_16xb256-coslr-200e_in1k_20220509_213052.log.json)                                 |
|                                                                                                                    | [byol_resnet50_8xb32-accum16-coslr-300e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-300e_in1k.py)                       | [model](https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_8xb32-accum16-coslr-300e_in1k_20220225-a0daa54a.pth) \| [log](https://download.openmmlab.com/mmselfsup/byol/byol_resnet50_8xb32-accum16-coslr-300e_in1k_20220210_095852.log.json)                     |
| [SwAV](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/README.md)                         | [swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py)     | [model](https://download.openmmlab.com/mmselfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96_20220225-0497dd5d.pth) \| [log](https://download.openmmlab.com/mmselfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96_20220211_061131.log.json)   |
| [DenseCL](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/densecl/README.md)                   | [densecl_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py)                              | [model](https://download.openmmlab.com/mmselfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k_20220225-8c7808fe.pth) \| [log](https://download.openmmlab.com/mmselfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k_20220215_041207.log.json)                         |
| [SimSiam](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/README.md)                   | [simsiam_resnet50_8xb32-coslr-100e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)                              | [model](https://download.openmmlab.com/mmselfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k_20220225-68a88ad8.pth) \| [log](https://download.openmmlab.com/mmselfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k_20220210_195405.log.json)                         |
|                                                                                                                    | [simsiam_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)                              | [model](https://download.openmmlab.com/mmselfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k_20220225-2f488143.pth) \| [log](https://download.openmmlab.com/mmselfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k_20220210_195402.log.json)                         |
| [BarlowTwins](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/barlowtwins/README.md)           | [barlowtwins_resnet50_8xb256-coslr-300e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k.py)                | [model](https://download.openmmlab.com/mmselfsup/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k_20220419-5ae15f89.pth) \| [log](https://download.openmmlab.com/mmselfsup/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k_20220413_111555.log.json)       |
| [MoCo v3](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov3/README.md)                    | [mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov3/mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224.py) | [model](https://download.openmmlab.com/mmselfsup/moco/mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224_20220225-e31238dd.pth) \| [log](https://download.openmmlab.com/mmselfsup/moco/mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224_20220222_160222.log.json) |
| [InterCLR](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/interclr/README.md)                 | [interclr-moco_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/interclr/interclr-moco_resnet50_8xb32-coslr-200e_in1k.py)                 | [model](https://download.openmmlab.com/mmselfsup/interclr/interclr-moco_resnet50_8xb32-coslr-200e_in1k_20221206-38f8fdaf.pth) \| [log](https://download.openmmlab.com/mmselfsup/interclr/interclr-moco_resnet50_8xb32-coslr-200e_in1k_20221103_085100.log.json)           |
| [MAE](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mae/README.md)                           | [mae_vit-base-p16_8xb512-coslr-400e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e-fp16_in1k.py)                           | [model](https://download.openmmlab.com/mmselfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k-224_20220223-85be947b.pth) \| [log](https://download.openmmlab.com/mmselfsup/mae/mae_vit-base-p16_8xb512-coslr-300e_in1k-224_20220210_140925.log.json)                       |
| [SimMIM](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simmim/README.md)                     | [simmim_swin-base_16xb128-coslr-100e_in1k-192](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simmim/simmim_swin-base_16xb128-coslr-100e_in1k-192.py)                   | [model](https://download.openmmlab.com/mmselfsup/simmim/simmim_swin-base_16xb128-coslr-100e_in1k-192_20220316-1d090125.pth) \| [log](https://download.openmmlab.com/mmselfsup/simmim/simmim_swin-base_16xb128-coslr-100e_in1k-192_20220316-1d090125.log.json)             |
| [CAE](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/cae/README.md)                           | [cae_vit-base-p16_8xb256-fp16-coslr-300e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/cae/cae_vit-base-p16_8xb256-fp16-coslr-300e_in1k.py)                      | [model](https://download.openmmlab.com/mmselfsup/cae/cae_vit-base-p16_16xb256-coslr-300e_in1k-224_20220427-4c786349.pth) \| [log](https://download.openmmlab.com/mmselfsup/cae/cae_vit-base-p16_16xb256-coslr-300e_in1k-224_20220427-4c786349.log.json)                   |
| [MaskFeat](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/maskfeat/README.md)                 | [maskfeat_vit-base-p16_8xb256-coslr-300e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/maskfeat/maskfeat_vit-base-p16_8xb256-coslr-300e_in1k.py)                 | [model](https://download.openmmlab.com/mmselfsup/maskfeat/maskfeat_vit-base-p16_8xb256-coslr-300e_in1k_20220913-591d4c4b.pth) \| [log](https://download.openmmlab.com/mmselfsup/maskfeat/maskfeat_vit-base-p16_8xb256-coslr-300e_in1k_20220829_225552.log.json)           |

Remarks:

- The training details are recorded in the config names.

- You can click algorithm name to obtain more information.
>>>>>>> upstream/master

## Benchmarks

### ImageNet

ImageNet has multiple versions, but the most commonly used one is ILSVRC 2012. The classification results below are reported by linear evaluation or fine-tuning with pre-trained weights provided by various algorithms.

<<<<<<< HEAD
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
      <td><a href='https://github.com/open-mmlab/mmselfsup/blob/dev-1.x/configs/benchmarks/classification/imagenet/swin-base_ft-8xb256-coslr-100e_in1k-192.py'>config</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/swin-base_ft-8xb256-coslr-100e_in1k/swin-base_ft-8xb256-coslr-100e_in1k_20220829-9cf23aa1.pth'>model</a> | <a href='https://download.openmmlab.com/mmselfsup/1.x/simmim/simmim_swin-base_8xb256-amp-coslr-100e_in1k-192/swin-base_ft-8xb256-coslr-100e_in1k/swin-base_ft-8xb256-coslr-100e_in1k_20220829_001452.json'>log</a></td>
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
</tbody>
</table>
=======
If not specified, we use linear evaluation setting from [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf) as default. Other settings are mentioned in Remarks.

| Algorithm           | Config                                                                                                                                                                                       | Remarks                    | Top-1 (%) |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | --------- |
| Relative Location   | [relative-loc_resnet50_8xb64-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)               |                            | 38.78     |
| Rotation Prediction | [rotation-pred_resnet50_8xb16-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py)            |                            | 48.12     |
| DeepCluster         | [deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/deepcluster/deepcluster-sobel_resnet50_8xb64-steplr-200e_in1k.py) |                            | 46.92     |
| NPID                | [npid_resnet50_8xb32-steplr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                                     |                            | 58.97     |
| ODC                 | [odc_resnet50_8xb64-steplr-440e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/odc/odc_resnet50_8xb64-steplr-440e_in1k.py)                                        |                            | 53.43     |
| SimCLR              | [simclr_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)                                 | SimSiam paper setting      | 62.56     |
|                     | [simclr_resnet50_16xb256-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simclr/simclr_resnet50_16xb256-coslr-200e_in1k.py)                             | SimSiam paper setting      | 66.66     |
| MoCo v2             | [mocov2_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                                 |                            | 67.58     |
| BYOL                | [byol_resnet50_8xb32-accum16-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)                       | SimSiam paper setting      | 71.72     |
|                     | [byol_resnet50_16xb256-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_16xb256-coslr-200e_in1k.py)                                   | SimSiam paper setting      | 71.88     |
|                     | [byol_resnet50_8xb32-accum16-coslr-300e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-300e_in1k.py)                       | SimSiam paper setting      | 72.93     |
| SwAV                | [swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py)     | SwAV paper setting         | 70.47     |
| DenseCL             | [densecl_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py)                              |                            | 63.62     |
| SimSiam             | [simsiam_resnet50_8xb32-coslr-100e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)                              | SimSiam paper setting      | 68.28     |
|                     | [simsiam_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)                              | SimSiam paper setting      | 69.84     |
| Barlow Twins        | [barlowtwins_resnet50_8xb256-coslr-300e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k.py)                | Barlow Twins paper setting | 71.66     |
| MoCo v3             | [mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov3/mocov3_vit-small-p16_32xb128-fp16-coslr-300e_in1k-224.py) | MoCo v3 paper setting      | 73.19     |
| InterCLR            | [interclr-moco_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/interclr/interclr-moco_resnet50_8xb32-coslr-200e_in1k.py)                 |                            | 68.04     |

### ImageNet Fine-tuning

| Algorithm | Config                                                                                                                                                                                                 | Remarks | Top-1 (%) |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- | --------- |
| MAE       | [mae_vit-base-p16_8xb512-coslr-400e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mae/mae_vit-base-p16_8xb512-coslr-400e_in1k.py)                                          |         | 83.1      |
| SimMIM    | [simmim_swin-base_16xb128-coslr-100e_in1k-192](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simmim/simmim_swin-base_16xb128-coslr-100e_in1k-192.py)                             |         | 82.9      |
| CAE       | [cae_vit-base-p16_8xb256-fp16-coslr-300e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/cae/cae_vit-base-p16_8xb256-fp16-coslr-300e_in1k.py)                                |         | 83.2      |
| MaskFeat  | [maskfeat_vit-base-p16_8xb256-fp16-coslr-300e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/classification/imagenet/maskfeat_vit-base-p16_ft-8xb512-coslr-100e_in1k.py) |         | 83.5      |

### COCO17 Object Detection and Instance Segmentation

In COCO17 object detection and instance segmentation task, we choose the evaluation protocol from [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf), with Mask-RCNN FPN architecture. The results below are fine-tuned with the same [config](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x_coco.py).

| Algorithm           | Config                                                                                                                                                                                   | mAP (Box) | mAP (Mask) |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---------- |
| Relative Location   | [relative-loc_resnet50_8xb64-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)           | 37.5      | 33.7       |
| Rotation Prediction | [rotation-pred_resnet50_8xb16-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py)        | 37.9      | 34.2       |
| NPID                | [npid_resnet50_8xb32-steplr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                                 | 38.5      | 34.6       |
| SimCLR              | [simclr_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)                             | 38.7      | 34.9       |
| MoCo v2             | [mocov2_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                             | 40.2      | 36.1       |
| BYOL                | [byol_resnet50_8xb32-accum16-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)                   | 40.9      | 36.8       |
| SwAV                | [swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py) | 40.2      | 36.3       |
| SimSiam             | [simsiam_resnet50_8xb32-coslr-100e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)                          | 38.6      | 34.6       |
|                     | [simsiam_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)                          | 38.8      | 34.9       |

### Pascal VOC12 Aug Semantic Segmentation

In Pascal VOC12 Aug semantic segmentation task, we choose the evaluation protocol from [MMSeg](https://github.com/open-mmlab/mmsegmentation), with FCN architecture. The results below are fine-tuned with the same [config](https://github.com/open-mmlab/mmselfsup/blob/master/configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py).

| Algorithm           | Config                                                                                                                                                                                   | mIOU  |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| Relative Location   | [relative-loc_resnet50_8xb64-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/relative_loc/relative-loc_resnet50_8xb64-steplr-70e_in1k.py)           | 63.49 |
| Rotation Prediction | [rotation-pred_resnet50_8xb16-steplr-70e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/rotation_pred/rotation-pred_resnet50_8xb16-steplr-70e_in1k.py)        | 64.31 |
| NPID                | [npid_resnet50_8xb32-steplr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/npid/npid_resnet50_8xb32-steplr-200e_in1k.py)                                 | 65.45 |
| SimCLR              | [simclr_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simclr/simclr_resnet50_8xb32-coslr-200e_in1k.py)                             | 64.03 |
| MoCo v2             | [mocov2_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/mocov2/mocov2_resnet50_8xb32-coslr-200e_in1k.py)                             | 67.55 |
| BYOL                | [byol_resnet50_8xb32-accum16-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/byol/byol_resnet50_8xb32-accum16-coslr-200e_in1k.py)                   | 67.16 |
| SwAV                | [swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96.py) | 63.73 |
| DenseCL             | [densecl_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/densecl/densecl_resnet50_8xb32-coslr-200e_in1k.py)                          | 69.47 |
| SimSiam             | [simsiam_resnet50_8xb32-coslr-100e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-100e_in1k.py)                          | 48.35 |
|                     | [simsiam_resnet50_8xb32-coslr-200e_in1k](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/simsiam/simsiam_resnet50_8xb32-coslr-200e_in1k.py)                          | 46.27 |
>>>>>>> upstream/master
