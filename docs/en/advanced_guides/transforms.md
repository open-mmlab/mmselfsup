# Transforms

- [Transforms]()
  - [Overview of transforms](#overview-of-transforms)
  - [Introduction of `MultiView`](#introduction-of-multiview)
  - [Introduction of `PackSelfSupInputs`](#introduction-of-packselfsupinputs)


## Overview of transforms
We have introduced how to build a `Pipeline` in [add_transforms](./add_transforms.md). A `Pipeline` contains a series of
`transforms`. There are three main categories of `transforms` in MMSelfSup:
1. Transforms about processing the data. The unique transforms in MMSelfSup are defined in [processing.py](../../../mmselfsup/datasets/transforms/processing.py), e.g. `RandomCrop`, `RandomResizedCrop` and `RandomGaussianBlur`.
We may also use some transforms from other repositories, e.g. `LoadImageFromFile` from MMCV.
2. The transform wrapper for multiple views of an image. It is defined in [wrappers.py](../../../mmselfsup/datasets/transforms/wrappers.py).
3. The transform to pack data into the format compatible with the inputs of algorithm. It is defined in [formatting.py](../../../mmselfsup/datasets/transforms/formatting.py).

## Introduction of `MultiView`


## Introduction of `PackSelfSupInputs`
