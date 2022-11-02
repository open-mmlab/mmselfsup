# Tutorial 2: Prepare Datasets

MMSelfSup supports multiple datasets. Please follow the corresponding guidelines for data preparation. It is recommended to symlink your dataset root to `$MMSELFSUP/data`. If your folder structure is different, you may need to change the corresponding paths in config files.

- [Tutorial 2: Prepare Datasets](#tutorial-2-prepare-datasets)
  - [Prepare ImageNet](#prepare-imagenet)
  - [Prepare Place205](#prepare-place205)
  - [Prepare iNaturalist2018](#prepare-inaturalist2018)
  - [Prepare PASCAL VOC](#prepare-pascal-voc)
  - [Prepare CIFAR10](#prepare-cifar10)
  - [Prepare datasets for detection and segmentation](#prepare-datasets-for-detection-and-segmentation)
    - [Detection](#detection)
    - [Segmentation](#segmentation)

```
mmselfsup
├── mmselfsup
├── tools
├── configs
├── docs
├── data
│   ├── imagenet
│   │   ├── meta
│   │   ├── train
│   │   ├── val
│   ├── places205
│   │   ├── meta
│   │   ├── train
│   │   ├── val
│   ├── inaturalist2018
│   │   ├── meta
│   │   ├── train
│   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2007
│   ├── cifar
│   │   ├── cifar-10-batches-py

```

## Prepare ImageNet

For ImageNet, it has multiple versions, but the most commonly used one is [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/). It can be accessed with the following steps:

1. Register an account and login to the [download page](http://www.image-net.org/download-images)
2. Find download links for ILSVRC2012 and download the following two files
   - ILSVRC2012_img_train.tar (~138GB)
   - ILSVRC2012_img_val.tar (~6.3GB)
3. Untar the downloaded files
4. Download meta data using this [script](https://github.com/BVLC/caffe/blob/master/data/ilsvrc12/get_ilsvrc_aux.sh)

## Prepare Place205

For Places205, you need to:

1. Register an account and login to the [download page](http://places.csail.mit.edu/downloadData.html)
2. Download the resized images and the image list of train set and validation set of Places205
3. Untar the downloaded files

## Prepare iNaturalist2018

For iNaturalist2018, you need to:

1. Download the training and validation images and annotations from the [download page](https://github.com/visipedia/inat_comp/tree/master/2018)
2. Untar the downloaded files
3. Convert the original json annotation format to the list format using the script `tools/dataset_converters/convert_inaturalist.py`

## Prepare PASCAL VOC

Assuming that you usually store datasets in `$YOUR_DATA_ROOT`. The following command will automatically download PASCAL VOC 2007 into `$YOUR_DATA_ROOT`, prepare the required files, create a folder `data` under `$MMSELFSUP` and make a symlink `VOCdevkit`.

```shell
bash tools/dataset_converters/prepare_voc07_cls.sh $YOUR_DATA_ROOT
```

## Prepare CIFAR10

`MMSelfSup` uses [`CIFAR10`](https://github.com/open-mmlab/mmclassification/blob/1.x/mmcls/datasets/cifar.py) implemented by `MMClassification`. In addition, `MMClassification` supports automatic download of the `CIFAR10` dataset, you just need to specify the download folder in the `data_root` field. And specify `test_mode=False` / `test_mode=True` to use the training or test dataset. For more details, please refer to [docs](https://github.com/open-mmlab/mmclassification/blob/1.x/docs/en/user_guides/dataset_prepare.md#cifar) in `MMClassification`.

## Prepare datasets for detection and segmentation

### Detection

To prepare COCO, VOC2007 and VOC2012 for detection, you can refer to [mmdetection](https://github.com/open-mmlab/mmdetection/blob/dev-3.x/docs/en/1_exist_data_model.md).

### Segmentation

To prepare VOC2012AUG and Cityscapes for segmentation, you can refer to [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/blob/dev-1.x/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets)
