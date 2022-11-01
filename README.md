<div align="center">
  <img src="./resources/mmselfsup_logo.png" width="500"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmselfsup)](https://pypi.org/project/mmselfsup)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmselfsup.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmselfsup/workflows/build/badge.svg)](https://github.com/open-mmlab/mmselfsup/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmselfsup/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmselfsup)
[![license](https://img.shields.io/github/license/open-mmlab/mmselfsup.svg)](https://github.com/open-mmlab/mmselfsup/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmselfsup.svg)](https://github.com/open-mmlab/mmselfsup/issues)

[📘Documentation](https://mmselfsup.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmselfsup.readthedocs.io/en/latest/install.html) |
[👀Model Zoo](https://github.com/open-mmlab/mmselfsup/blob/master/docs/en/model_zoo.md) |
[🆕Update News](https://mmselfsup.readthedocs.io/en/latest/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmselfsup/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction

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

## What's New

### Preview of 1.x version

A brand new version of **MMSelfSup v1.0.0rc1** was released in 01/09/2022:

Highlights of the new version:

- Based on [MMEngine](https://github.com/open-mmlab/mmengine) and [MMCV](https://github.com/open-mmlab/mmcv/tree/2.x).
- Released with refactor.
- Refine all [documents](https://mmselfsup.readthedocs.io/en/1.x/).
- Support `MAE`, `SimMIM`, `MoCoV3` with different pre-training epochs and backbones of different scales.
- More concise API.
- More powerful data pipeline.
- Higher accurcy for some algorithms.

Find more new features in [1.x branch](https://github.com/open-mmlab/mmselfsup/tree/1.x). Issues and PRs are welcome!

### Stable version

MMSelfSup **v0.10.1** was released in 01/11/2022.

Highlights of the new version:

- Support MaskFeat
- Update issue form
- Fix some typo in documents

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

Differences between MMSelfSup and OpenSelfSup codebases can be found in [compatibility.md](docs/en/compatibility.md).

## Installation

MMSelfSup relies on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMClassification](https://github.com/open-mmlab/mmclassification).

Please refer to [install.md](docs/en/install.md) for more detailed instruction.

## Get Started

Please refer to [prepare_data.md](docs/en/prepare_data.md) for dataset preparation, [get_started.md](docs/en/get_started.md) for the basic usage and [benchmarks.md](docs/en/tutorials/6_benchmarks.md) for running benchmarks.

We also provides more detailed tutorials:

- [config](docs/en/tutorials/0_config.md)
- [add new dataset](docs/en/tutorials/1_new_dataset.md)
- [data pipeline](docs/en/tutorials/2_data_pipeline.md)
- [add new module](docs/en/tutorials/3_new_module.md)
- [customize schedules](docs/en/tutorials/4_schedule.md)
- [customize runtime](docs/en/tutorials/5_runtime.md)

Besides, we provide [colab tutorial](https://github.com/open-mmlab/mmselfsup/blob/master/demo/mmselfsup_colab_tutorial.ipynb) for basic usage.

Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.

## Model Zoo

Please refer to [model_zoo.md](docs/en/model_zoo.md) for a comprehensive set of pre-trained models and benchmarks.

Supported algorithms:

- [x] [Relative Location (ICCV'2015)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/relative_loc)
- [x] [Rotation Prediction (ICLR'2018)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/rotation_pred)
- [x] [DeepCluster (ECCV'2018)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/deepcluster)
- [x] [NPID (CVPR'2018)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/npid)
- [x] [ODC (CVPR'2020)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/odc)
- [x] [MoCo v1 (CVPR'2020)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/mocov1)
- [x] [SimCLR (ICML'2020)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/simclr)
- [x] [MoCo v2 (ArXiv'2020)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/byol)
- [x] [BYOL (NeurIPS'2020)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/mocov2)
- [x] [SwAV (NeurIPS'2020)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/swav)
- [x] [DenseCL (CVPR'2021)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/densecl)
- [x] [SimSiam (CVPR'2021)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/simsiam)
- [x] [Barlow Twins (ICML'2021)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/barlowtwins)
- [x] [MoCo v3 (ICCV'2021)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/mocov3)
- [x] [MAE (CVPR'2022)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/mae)
- [x] [SimMIM (CVPR'2022)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/simmim)
- [x] [MaskFeat (CVPR'2022)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/maskfeat)
- [x] [CAE (ArXiv'2022)](https://github.com/open-mmlab/mmselfsup/tree/master/configs/selfsup/cae)

More algorithms are in our plan.

## Benchmark

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

## Contributing

We appreciate all contributions improving MMSelfSup. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for more details about the contributing guideline.

## Acknowledgement

MMSelfSup is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new algorithms.

MMSelfSup originates from OpenSelfSup, and we appreciate all early contributions made to OpenSelfSup. A few contributors are listed here: Xiaohang Zhan ([@XiaohangZhan](http://github.com/XiaohangZhan)), Jiahao Xie ([@Jiahao000](https://github.com/Jiahao000)), Enze Xie ([@xieenze](https://github.com/xieenze)), Xiangxiang Chu ([@cxxgtxy](https://github.com/cxxgtxy)), Zijian He ([@scnuhealthy](https://github.com/scnuhealthy)).

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

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
