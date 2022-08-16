# Segmentation

- [Segmentation](#segmentation)
  - [Train](#train)

For semantic segmentation task, we use MMSegmentation. First, make sure you have installed [MIM](https://github.com/open-mmlab/mim), which is also a project of OpenMMLab.

```shell
pip install openmim
```

It is very easy to install the package.

Besides, please refer to MMSeg for [installation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md) and [data preparation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md#prepare-datasets).

## Train

After installation, you can run MMSeg with simple command.

```shell
# distributed version
bash tools/benchmarks/mmsegmentation/mim_dist_train.sh ${CONFIG} ${PRETRAIN} ${GPUS}

# slurm version
bash tools/benchmarks/mmsegmentation/mim_slurm_train.sh ${PARTITION} ${CONFIG} ${PRETRAIN}
```

Remarks:

- `CONFIG`: Use config files under `configs/benchmarks/mmsegmentation/` or write your own config files
- `PRETRAIN`: the pre-trained model file.
