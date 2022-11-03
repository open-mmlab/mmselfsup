# Segmentation

- [Segmentation](#segmentation)
  - [Train](#train)
  - [Test](#test)

For semantic segmentation task, we use MMSegmentation. First, make sure you have installed [MIM](https://github.com/open-mmlab/mim), which is also a project of OpenMMLab.

```shell
pip install openmim
mim install 'mmsegmentation>=1.0.0rc0'
```

It is very easy to install the package.

Besides, please refer to MMSegmentation for [installation](https://mmsegmentation.readthedocs.io/en/dev-1.x/get_started.html) and [data preparation](https://mmsegmentation.readthedocs.io/en/dev-1.x/user_guides/2_dataset_prepare.html).

## Train

After installation, you can run MMSeg with simple command.

```shell
# distributed version
bash tools/benchmarks/mmsegmentation/mim_dist_train.sh ${CONFIG} ${PRETRAIN} ${GPUS}

# slurm version
bash tools/benchmarks/mmsegmentation/mim_slurm_train.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
```

Remarks:

- `CONFIG`: Use config files under `configs/benchmarks/mmsegmentation/`. Since repositories of OpenMMLab have support referring config files across different
  repositories, we can easily leverage the configs from MMSegmentation like:

```shell
_base_ = 'mmseg::fcn/fcn_r50-d8_4xb2-40k_cityscapes-769x769.py'
```

Writing your config files from scratch is also supported.

- `PRETRAIN`: the pre-trained model file.
- `GPUS`: The number of GPUs that you want to use to train. We adopt 4 GPUs for segmentation tasks by default.

Example:

```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_train.sh \
configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 4
```

## Test

After training, you can also run the command below to test your model.

```shell
# distributed version
bash tools/benchmarks/mmsegmentation/mim_dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS}

# slurm version
bash tools/benchmarks/mmsegmentation/mim_slurm_test.sh ${PARTITION} ${CONFIG} ${CHECKPOINT}
```

Remarks:

- `CHECKPOINT`: The well-trained segmentation model that you want to test.

Example:

```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_test.sh \
configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_4xb4-20k_voc12aug-512x512.py \
https://download.openmmlab.com/mmselfsup/1.x/byol/byol_resnet50_16xb256-coslr-200e_in1k/byol_resnet50_16xb256-coslr-200e_in1k_20220825-de817331.pth 4
```
