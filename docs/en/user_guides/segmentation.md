# Segmentation

- [Segmentation](#segmentation)
  - [Train](#train)
  - [Test](#test)

For semantic segmentation task, we use MMSegmentation. First, make sure you have installed [MIM](https://github.com/open-mmlab/mim), which is also a project of OpenMMLab.

```shell
pip install openmim
```

It is very easy to install the package.

Besides, please refer to MMSegmentation for [installation](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/get_started.md) and [data preparation](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/user_guides/2_dataset_prepare.md).

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
_base_ = 'mmseg::fcn/fcn_r50-d8_769x769_40k_cityscapes.py'
```
Writing your config files from scratch is also supported.

- `PRETRAIN`: the pre-trained model file.
- `GPUS`: The number of GPUs that you want to use to train. We adopt 4 GPUs for segmentation tasks by default.

Example:
```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_train.sh \
configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py \
work_dir/byol_resnet50_8xb32-accum16-coslr-200e_in1k_20220225-5c8b2c2e.pth 4
```

## Test
After training, you can also run the command bellow to test your model.
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
configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py \
work_dir/iter_20000.pth 4
```
