<div align="center">
  <img src="./resources/mmselfsup_logo.png" width="500"/>

[![PyPI](https://img.shields.io/pypi/v/mmselfsup)](https://pypi.org/project/mmselfsup)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmselfsup.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmselfsup/workflows/build/badge.svg)](https://github.com/open-mmlab/mmselfsup/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmselfsup/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmselfsup)
[![license](https://img.shields.io/github/license/open-mmlab/mmselfsup.svg)](https://github.com/open-mmlab/mmselfsup/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmselfsup.svg)](https://github.com/open-mmlab/mmselfsup/issues)

</div>

## Introduction

English | [简体中文](README_zh-CN.md)

MMSelfSup is an open source self-supervised representation learning toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.5** or higher.


### Major features

- **Methods All in One**

  MMSelfsup provides state-of-the-art methods in self-supervised learning. For comprehensive comparison in all benchmarks, most of the pre-training methods are under the same setting.

- **Modular Design**

  MMSelfSup follows a similar code architecture of OpenMMLab projects with modular design, which is flexible and convenient for users to build their own algorithms.

- **Standardized Benchmarks**

  MMSelfSup standardizes the benchmarks including logistic regression, SVM / Low-shot SVM from linearly probed features, semi-supervised classification, object detection and semantic segmentation.

- **Compatibility**

  Since MMSelfSup adopts similar design of modulars and interfaces as those in other OpenMMLab projects, it supports smooth evaluation on downstream tasks with other OpenMMLab projects like object detection and segmentation.


## License

This project is released under the [Apache 2.0 license](LICENSE).

## ChangeLog

MMSelfSup **v0.5.0** was released with refactor in 16/12/2021.

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

Differences between MMSelfSup and OpenSelfSup codebases can be found in [compatibility.md](docs/en/compatibility.md).

## Model Zoo and Benchmark

### Model Zoo
Please refer to [model_zoo.md](docs/en/model_zoo.md) for a comprehensive set of pre-trained models and benchmarks.

Supported algorithms:

- [x] [Relative Location (ICCV'2015)](https://arxiv.org/abs/1505.05192)
- [x] [Rotation Prediction (ICLR'2018)](https://arxiv.org/abs/1803.07728)
- [x] [DeepCLuster (ECCV'2018)](https://arxiv.org/abs/1807.05520)
- [x] [NPID (CVPR'2018)](https://arxiv.org/abs/1805.01978)
- [x] [ODC (CVPR'2020)](https://arxiv.org/abs/2006.10645)
- [x] [MoCo v1 (CVPR'2020)](https://arxiv.org/abs/1911.05722)
- [x] [SimCLR (ICML'2020)](https://arxiv.org/abs/2002.05709)
- [x] [MoCo v2 (ArXiv'2020)](https://arxiv.org/abs/2003.04297)
- [x] [BYOL (NeurIPS'2020)](https://arxiv.org/abs/2006.07733)
- [x] [SwAV (NeurIPS'2020)](https://arxiv.org/abs/2006.09882)
- [x] [DenseCL (CVPR'2021)](https://arxiv.org/abs/2011.09157)
- [x] [SimSiam (CVPR'2021)](https://arxiv.org/abs/2011.10566)

More algorithms are in our plan.

### Benchmark

  | Benchmarks                                         | Setting                                                                                                                                                              |
  | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | ImageNet Linear Classification (Multi-head)        | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
  | ImageNet Linear Classification (Last)              |                                                                                                                                                                      |
  | ImageNet Semi-Sup Classification                   |                                                                                                                                                                      |
  | Places205 Linear Classification (Multi-head)       | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
  | iNaturalist2018 Linear Classification (Multi-head) | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
  | PASCAL VOC07 SVM                                   | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
  | PASCAL VOC07 Low-shot SVM                          | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
  | PASCAL VOC07+12 Object Detection                   | [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)               |
  | COCO17 Object Detection                            | [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)               |
  | Cityscapes Segmentation                            | [MMSeg](configs/benchmarks/mmsegmentation/cityscapes/fcn_r50-d8_769x769_40k_cityscapes.py)                                                                           |
  | PASCAL VOC12 Aug Segmentation                      | [MMSeg](configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py)                                                                               |

## Installation

Please refer to [install.md](docs/en/install.md) for installation and [prepare_data.md](docs/en/prepare_data.md) for dataset preparation.

## Get Started

Please see [getting_started.md](docs/en/getting_started.md) for the basic usage of MMSelfSup.

We also provides tutorials for more details:
- [config](docs/en/tutorials/0_config.md)
- [add new dataset](docs/en/tutorials/1_new_dataset.md)
- [data pipeline](docs/en/tutorials/2_data_pipeline.md)
- [add new module](docs/en/tutorials/3_new_module.md)
- [customize schedules](docs/en/tutorials/4_schedule.md)
- [customize runtime](docs/en/tutorials/5_runtime.md)
- [benchmarks](docs/en/tutorials/6_benchmarks.md)

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```bibteX
@misc{mmselfsup2021,
    title={{MMSelfSup}: OpenMMLab Self-Supervised Learning Toolbox and Benchmark},
    author={MMSelfSup Contributors},
    howpublished={\url{https://github.com/open-mmlab/mmselfsup}},
    year={2021}
}
```

## Contributing

We appreciate all contributions improving MMSelfSup. Please refer to [CONTRIBUTING.md](docs/en/community/CONTRIBUTING.md) for more details about the contributing guideline.

## Acknowledgement

Remarks:

- MMSelfSup originates from OpenSelfSup, and we appreciate all early contributions made to OpenSelfSup. A few contributors are listed here: Xiaohang Zhan, Jiahao Xie, Enze Xie, Xiangxiang Chu, Zijian He.
- The implementation of MoCo and the detection benchmark borrow the code from [MoCo](https://github.com/facebookresearch/moco).
- The implementation of SwAV borrows the code from [SwAV](https://github.com/facebookresearch/swav).
- The SVM benchmark borrows the code from [fair_self_supervision_benchmark](https://github.com/facebookresearch/fair_self_supervision_benchmark).
- `mmselfsup/utils/clustering.py` is borrowed from [deepcluster](https://github.com/facebookresearch/deepcluster/blob/master/clustering.py).

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab toolbox for text detection, recognition and understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMlab toolkit for generative models.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab few shot learning toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
