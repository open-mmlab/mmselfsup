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
[![license](https://img.shields.io/github/license/open-mmlab/mmselfsup.svg)](https://github.com/open-mmlab/mmselfsup/blob/main/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmselfsup.svg)](https://github.com/open-mmlab/mmselfsup/issues)

[📘Documentation](https://mmselfsup.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmselfsup.readthedocs.io/en/latest/get_started.html) |
[👀Model Zoo](https://mmselfsup.readthedocs.io/en/latest/model_zoo.html) |
[🆕Update News](https://mmselfsup.readthedocs.io/en/latest/notes/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmselfsup/issues/new/choose)

<img src="https://user-images.githubusercontent.com/36138628/230306412-43a5f316-bd54-4d2a-b196-210656e74683.png" width="500"/>

🌟 MMPreTrain is a newly upgraded open-source framework for visual pre-training. It has set out to provide multiple powerful pre-trained backbones and support different pre-training strategies.

:point_right: **MMPreTrain 1.0 branch is in trial, welcome every to [try it](https://github.com/open-mmlab/mmclassification) and [discuss with us](https://github.com/open-mmlab/mmclassification/discussions)!** :point_left:

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219255827-67c1a27f-f8c5-46a9-811d-5e57448c61d1.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.gg/raweFPmdzG" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://space.bilibili.com/1293512903" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026751-d7d14cce-a7c9-4e82-9942-8375fca65b99.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.zhihu.com/people/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/219026120-ba71e48b-6e94-4bd4-b4e9-b7d175b5e362.png" width="3%" alt="" /></a>
</div>

## Introduction

MMSelfSup is an open source self-supervised representation learning toolbox based on PyTorch. It is a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.8** or higher.

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

**MMSelfSup v1.0.0 was released based on `main` branch. Please refer to [Migration Guide](https://mmselfsup.readthedocs.io/en/latest/migration.html) for more details.**

MMSelfSup **v1.0.0** was released in 06/04/2023.

- Support `PixMIM`.
- Support `DINO` in `projects/dino/`.
- Refactor file io interface.
- Refine documentations.

MMSelfSup **v1.0.0rc6** was released in 10/02/2023.

- Support `MaskFeat` with video dataset in `projects/maskfeat_video/`
- Translate documentation to Chinese.

MMSelfSup **v1.0.0rc5** was released in 30/12/2022.

- Support `BEiT v2`, `MixMIM`, `EVA`.
- Support `ShapeBias` for model analysis
- Add Solution of FGIA ACCV 2022 (1st Place)
- Refactor t-SNE

Please refer to [Changelog](https://mmselfsup.readthedocs.io/en/latest/notes/changelog.html) for details and release history.

Differences between MMSelfSup 1.x and 0.x can be found in [Migration](https://mmselfsup.readthedocs.io/en/latest/migration.html).

## Installation

MMSelfSup depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv), [MMEngine](https://github.com/open-mmlab/mmengine) and [MMClassification](https://github.com/open-mmlab/mmclassification).

Please refer to [Installation](https://mmselfsup.readthedocs.io/en/latest/get_started.html) for more detailed instruction.

## Get Started

For tutorials, we provide [User Guides](https://mmselfsup.readthedocs.io/en/latest/user_guides/index.html) for basic usage:

Pretrain

- [Config](https://mmselfsup.readthedocs.io/en/latest/user_guides/1_config.html)
- [Prepare Dataset](https://mmselfsup.readthedocs.io/en/latest/user_guides/2_dataset_prepare.html)
- [Pretrain with Existing Models](https://mmselfsup.readthedocs.io/en/latest/user_guides/3_pretrain.html)

Downetream Tasks

- [Classification](https://mmselfsup.readthedocs.io/en/latest/user_guides/classification.html)
- [Detection](https://mmselfsup.readthedocs.io/en/latest/user_guides/detection.html)
- [Segmentation](https://mmselfsup.readthedocs.io/en/latest/user_guides/segmentation.html)

Useful Tools

- [Visualization](https://mmselfsup.readthedocs.io/en/latest/user_guides/visualization.html)
- [Analysis Tools](https://mmselfsup.readthedocs.io/en/latest/user_guides/analysis_tools.html)

[Advanced Guides](https://mmselfsup.readthedocs.io/en/latest/advanced_guides/index.html) and [Colab Tutorials](https://github.com/open-mmlab/mmselfsup/blob/main/demo/mmselfsup_colab_tutorial.ipynb) are also provided.

Please refer to [FAQ](https://mmselfsup.readthedocs.io/en/latest/notes/faq.html) for frequently asked questions.

## Model Zoo

Please refer to [Model Zoo.md](https://mmselfsup.readthedocs.io/en/latest/model_zoo.html) for a comprehensive set of pre-trained models and benchmarks.

Supported algorithms:

- [x] [Relative Location (ICCV'2015)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/relavive_loc)
- [x] [Rotation Prediction (ICLR'2018)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/rotation_pred)
- [x] [DeepCluster (ECCV'2018)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/deepcluster)
- [x] [NPID (CVPR'2018)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/npid)
- [x] [ODC (CVPR'2020)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/odc)
- [x] [MoCo v1 (CVPR'2020)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/mocov1)
- [x] [SimCLR (ICML'2020)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/simclr)
- [x] [MoCo v2 (arXiv'2020)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/mocov2)
- [x] [BYOL (NeurIPS'2020)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/byol)
- [x] [SwAV (NeurIPS'2020)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/swav)
- [x] [DenseCL (CVPR'2021)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/densecl)
- [x] [SimSiam (CVPR'2021)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/simsiam)
- [x] [Barlow Twins (ICML'2021)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/barlowtwins)
- [x] [MoCo v3 (ICCV'2021)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/mocov3)
- [x] [BEiT (ICLR'2022)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/beit)
- [x] [MAE (CVPR'2022)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/mae)
- [x] [SimMIM (CVPR'2022)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/simmim)
- [x] [MaskFeat (CVPR'2022)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/maskfeat)
- [x] [CAE (arXiv'2022)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/cae)
- [x] [MILAN (arXiv'2022)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/milan)
- [x] [BEiT v2 (arXiv'2022)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/beitv2)
- [x] [EVA (CVPR'2023)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/eva)
- [x] [MixMIM (ArXiv'2022)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/mixmim)
- [x] [PixMIM (ArXiv'2023)](https://github.com/open-mmlab/mmselfsup/tree/main/configs/selfsup/pixmim)

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

We appreciate all contributions improving MMSelfSup. Please refer to [Contribution Guides](https://mmselfsup.readthedocs.io/en/latest/notes/contribution_guide.html) for more details about the contributing guideline.

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

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab foundational library for training deep learning models.
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMEval](https://github.com/open-mmlab/mmeval): A unified evaluation library for multiple machine learning libraries.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark.
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
