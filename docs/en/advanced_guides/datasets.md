# Datasets

- [Datasets](#datasets)
  - [Datasets](#datasets-1)
    - [Refactor your datasets](#refactor-your-datasets)
    - [Use datasets from other MM-repos in your config](#use-datasets-from-other-mm-repos-in-your-config)
  - [Samplers](#samplers)
  - [Transforms](#transforms)

The `datasets` folder under `mmselfsup` contains all kinds of modules, related to loading data.
It can be roughly split into three parts, namely,

- cutomized datasets to read images
- cutomized dataset samplers to read index before loading images
- data transforms, e.g. `RandomResizedCrop`, to augment data before feeding into models.

In this tutorial, we will explain the above three parts in details.

## Datasets

OpenMMLab provides a lot of off-the-shelf datasets, and all these datasets inherit the [BaseDataset](https://github.com/open-mmlab/mmengine/blob/429bb27972bee1a9f3095a4d5f6ac5c0b88ccf54/mmengine/dataset/base_dataset.py#L116)
implemented in [MMEngine](https://github.com/open-mmlab/mmengine). To have a full knowledge about all these functionalities implemented in
`BaseDataset`, we recommend interested readers to refer to the documents in [MMEngine](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/basedataset.html). `ImageNet`, `ADE20KDataset` and `CocoDataset` are the three commonly used datasets `MMSelfSup`. Before using them, you should refactor your local folder according to
the following format.

### Refactor your datasets

To use these existing datasets, you need to refactor your datasets
into following dataset format.

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
│   │
│   │── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │
│   │── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

For more details about the annotation files and the structure of each subfolder, you can consult [MMClassfication](https://github.com/open-mmlab/mmclassification),
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [MMDetection](https://github.com/open-mmlab/mmdetection).

### Use datasets from other MM-repos in your config

```python
# Use ImageNet dataset from MMClassification
# Use ImageNet in your dataloader
# For simplicity, we only provide the config related to importting ImageNet
# from MMClassification, instead of the full configuration for the dataloader.
# The ``mmcls`` prefix tells the ``Registry`` to search ``ImageNet`` in
# MMClassification
train_dataloader=dict(dataset=dict(type='mmcls.ImageNet', ...), ...)
```

```python
# Use ADE20KDataset dataset from MMSegmentation
# Use ADE20KDataset in your dataloader
# For simplicity, we only provide the config related to importting ADE20KDataset
# from MMSegmentation, instead of the full configuration for the dataloader.
# The ``mmseg`` prefix tells the ``Registry`` to search ``ADE20KDataset`` in
# MMSegmentation
train_dataloader=dict(dataset=dict(type='mmseg.ADE20KDataset', ...), ...)
```

```python
# Use CocoDataset in your dataloader
# For simplicity, we only provide the config related to importting CocoDataset
# from MMDetection, instead of the full configuration for the dataloader.
# The ``mmdet`` prefix tells the ``Registry`` to search ``CocoDataset`` in
# MMDetection
train_dataloader=dict(dataset=dict(type='mmdet.CocoDataset', ...), ...)
```

```python
# Use dataset in MMSelfSup, for example ``DeepClusterImageNet``
train_dataloader=dict(dataset=dict(type='DeepClusterImageNet', ...), ...)
```

Till now, we have introduced two key steps, in order to use existing datasets successfully. We hope you can
grasp the basic idea about how to use datasets in `MMSelfSup`. If you want to create you customized datasets, you can refer to
another useful document, [add_datasets](./add_datasets.md).

## Samplers

In pytorch, `Sampler` is used to sample the index of data before loading. `MMEngine` has already implemented `DefaultSampler` and
`InfiniteSampler`. In most situation, we can directly use them, instead of implementing customized sampler. But the `DeepClusterSampler` is a special case, in which we implement the unique index sampling logic. We recommend interested user to refer to the API doc for more details about this sampler. If you want to implement your customized sampler, you can follow `DeepClusterSampler`and implement it under the folder of `samplers`.

## Transforms

In short, `transform` refer to data augmentation in `MM-repos` and we compose a series of transforms into a list, called `pipeline`.
`MMCV` already provides some useful transforms, covering most of scenarios. But every `MM-repo` defines their own transforms, following
the [User Guide](https://github.com/open-mmlab/mmcv/blob/dev-2.x/docs/zh_cn/understand_mmcv/data_transform.md) in `MMCV`. Concretely, every
customized dataset: i) inherits [BaseTransform](https://github.com/open-mmlab/mmcv/blob/19a024155a0b710568c2faeae07dead2a5550392/mmcv/transforms/base.py#L6),
ii) overwrite the `transform` function and implement your key logic in it. In MMSelfSup, we implement these `transforms` below:

|                                                      class                                                      |
| :-------------------------------------------------------------------------------------------------------------: |
|                           [`PackSelfSupInputs`](mmselfsup.datasets.PackSelfSupInputs)                           |
|                           [`BEiTMaskGenerator`](mmselfsup.datasets.BEiTMaskGenerator)                           |
|                         [`SimMIMMaskGenerator`](mmselfsup.datasets.SimMIMMaskGenerator)                         |
|                                 [`ColorJitter`](mmselfsup.datasets.ColorJitter)                                 |
|                                  [`RandomCrop`](mmselfsup.datasets.RandomCrop)                                  |
|                          [`RandomGaussianBlur`](mmselfsup.datasets.RandomGaussianBlur)                          |
|                           [`RandomResizedCrop`](mmselfsup.datasets.RandomResizedCrop)                           |
| [`RandomResizedCropAndInterpolationWithTwoPic`](mmselfsup.datasets.RandomResizedCropAndInterpolationWithTwoPic) |
|                              [`RandomSolarize`](mmselfsup.datasets.RandomSolarize)                              |
|                          [`RotationWithLabels`](mmselfsup.datasets.RotationWithLabels)                          |
|                       [`RandomPatchWithLabels`](mmselfsup.datasets.RandomPatchWithLabels)                       |
|                              [`RandomRotation`](mmselfsup.datasets.RandomRotation)                              |

For interested users, you can refer to the API doc to have a full understanding of these transforms. Now, we have introduced
the basic concepts about transform. If you want to know how to use them in your config or implement your customed transforms,
you can refer to [transforms](./transforms.md) and [add_transforms](./add_transforms.md).
