# 准备数据集

MMSelfSup 支持多个数据集。请遵循相应的数据准备指南。建议将您的数据集根目录软链接到 `$MMSELFSUP/data`。如果您的文件夹结构不同，您可能需要更改配置文件中的相应路径。

- [准备 ImageNet 数据集](#%E5%87%86%E5%A4%87-imagenet-%E6%95%B0%E6%8D%AE%E9%9B%86)
- [准备 Places205 数据集](#%E5%87%86%E5%A4%87-places205-%E6%95%B0%E6%8D%AE%E9%9B%86)
- [准备 iNaturalist2018 数据集](#%E5%87%86%E5%A4%87-inaturalist2018-%E6%95%B0%E6%8D%AE%E9%9B%86)
- [准备 PASCAL VOC 数据集](#%E5%87%86%E5%A4%87-pascal-voc-%E6%95%B0%E6%8D%AE%E9%9B%86)
- [准备 CIFAR10 数据集](#%E5%87%86%E5%A4%87-cifar10-%E6%95%B0%E6%8D%AE%E9%9B%86)
- [准备检测和分割数据集](#%E5%87%86%E5%A4%87%E6%A3%80%E6%B5%8B%E5%92%8C%E5%88%86%E5%89%B2%E6%95%B0%E6%8D%AE%E9%9B%86)
  - [检测](#%E6%A3%80%E6%B5%8B)
  - [分割](#%E5%88%86%E5%89%B2)

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

## 准备 ImageNet 数据集

对于 ImageNet，它有多个版本，但最常用的是 [ILSVRC 2012](http://www.image-net.org/challenges/LSVRC/2012/)。可以通过以下步骤得到：

1. 注册账号并登录 [下载页面](http://www.image-net.org/download-images)
2. 找到 ILSVRC2012 的下载链接，下载以下两个文件
   - ILSVRC2012_img_train.tar (~138GB)
   - ILSVRC2012_img_val.tar (~6.3GB)
3. 解压下载的文件
4. 使用这个 [脚本](https://github.com/BVLC/caffe/blob/master/data/ilsvrc12/get_ilsvrc_aux.sh) 下载元数据

## 准备 Places205 数据集

对于 Places205，您需要：

1. 注册账号并登录 [下载页面](http://places.csail.mit.edu/downloadData.html)
2. 下载 Places205 经过缩放的图片以及训练集和验证集的图片列表
3. 解压下载的文件

## 准备 iNaturalist2018 数据集

对于 iNaturalist2018，您需要：

1. 从 [下载页面](https://github.com/visipedia/inat_comp/tree/master/2018) 下载训练集和验证集图像及标注
2. 解压下载的文件
3. 使用脚本 `tools/data_converters/convert_inaturalist.py` 将原来的 json 标注格式转换为列表格式

## 准备 PASCAL VOC 数据集

假设您通常将数据集存储在 `$YOUR_DATA_ROOT` 中。下面的命令会自动将 PASCAL VOC 2007 下载到 `$YOUR_DATA_ROOT` 中，准备好所需的文件，在 `$MMSELFSUP` 下创建一个文件夹 `data`，并制作一个软链接 `VOCdevkit`。

```shell
bash tools/data_converters/prepare_voc07_cls.sh $YOUR_DATA_ROOT
```

## 准备 CIFAR10 数据集

如果没有找到 CIFAR10 系统将会自动下载。此外，由 `MMSelfSup` 实现的 `dataset` 也会自动将 CIFAR10 转换为适当的格式。

## 准备检测和分割数据集

### 检测

您可以参考 [mmdet](https://github.com/open-mmlab/mmdetection/blob/master/docs/1_exist_data_model.md) 来准备 COCO，VOC2007 和 VOC2012 检测数据集。

### 分割

您可以参考 [mmseg](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md#prepare-datasets) 来准备 VOC2012AUG 和 Cityscapes 分割数据集。
