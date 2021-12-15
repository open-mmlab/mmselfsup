<div align="center">
  <img src="./resources/mmselfsup_logo.png" width="600"/>
</div>

[![PyPI](https://img.shields.io/pypi/v/mmselfsup)]()
[![docs](https://img.shields.io/badge/docs-latest-blue)]()
[![badge](https://github.com/open-mmlab/mmselfsup/workflows/build/badge.svg)]()
[![codecov](https://codecov.io/gh/open-mmlab/mmselfsup/branch/master/graph/badge.svg)]()
[![license](https://img.shields.io/github/license/open-mmlab/mmselfsup.svg)]()

[English](README.md) | 简体中文

## 介绍

`MMSelfSup` 是一个基于 PyTorch 实现的开源无监督表征学习工具箱，是 [OpenMMLab](https://openmmlab.com/) 项目成员之一。

主分支代码支持 **PyTorch 1.5** 及以上的版本。

### 主要特性

- **多方法集成**

  MMSelfSup 提供了多种前沿的自监督学习算法，大部分的自监督预训练学习都设置相同，以在基准中获得更加公平的比较。

- **模块化设计**

  MMSelfSup 遵照 OpenMMLab 项目一贯的设计理念，进行模块化设计，便于用户自定义实现自己的算法。

- **标准化的性能评测**

  MMSelfSup 拥有丰富的基准进行评估和测试，包括线性评估, 线性特征的 SVM / Low-shot SVM, 半监督分类, 目标检测和语义分割。

- **兼容性**

  兼容 OpenMMLab 各大算法库，拥有丰富的下游评测任务和预训练模型的应用。

## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE).

## 模型库和基准测试

### 模型库

请参考 [模型库](docs/model_zoo.md) 查看我们更加全面的模型基准结果。

目前已支持的算法:

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

更多的算法实现已经在我们的计划中。

### 基准测试

| 基准测试方法                                 | 参考设置                                                                                                                                                             |
| -------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ImageNet Linear Classification (Multi-head)  | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
| ImageNet Linear Classification               |                                                                                                                                                                      |
| ImageNet Semi-Sup Classification             |                                                                                                                                                                      |
| Places205 Linear Classification (Multi-head) | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
| iNaturalist 2018 Classification              | [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)               |
| PASCAL VOC07 SVM                             | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
| PASCAL VOC07 Low-shot SVM                    | [Goyal2019](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) |
| PASCAL VOC07+12 Object Detection             | [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)               |
| COCO17 Object Detection                      | [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)               |
| Cityscapes Segmentation                      | [MMSeg](configs/benchmarks/mmsegmentation/cityscapes/fcn_r50-d8_769x769_40k_cityscapes.py)                                                                           |
| PASCAL VOC12 Aug Segmentation                | [MMSeg](configs/benchmarks/mmsegmentation/voc12aug/fcn_r50-d8_512x512_20k_voc12aug.py)                                                                               |

## 安装

请参考 [安装文档](docs_zh-CN/install.md) 进行安装和参考 [准备数据](docs_zh-CN/prepare_data.md) 准备数据集。

## 快速入门

请参考 [入门指南](docs_zh-CN/getting_started.md) 获取 MMSelfSup 的基本使用方法.

我们也提供了更加全面的教程，包括:
- [配置文件](docs_zh-CN/tutorials/0_config.md)
- [添加数据集](docs_zh-CN/tutorials/1_new_dataset.md)
- [数据处理流](docs_zh-CN/tutorials/2_data_pipeline.md)
- [添加新模块](docs_zh-CN/tutorials/3_new_module.md)
- [自定义流程](docs_zh-CN/tutorials/4_schedule.md)
- [自定义运行](docs_zh-CN/tutorials/5_runtime.md)
- [基准测试](docs_zh-CN/tutorials/6_benchmarks.md)

## 参与贡献

我们非常欢迎任何有助于提升 MMSelfSup 的贡献，请参考 [贡献指南](docs_zh-CN/community/CONTRIBUTING.md) 来了解如何参与贡献。

## 致谢

MMSulfSup 是一款由不同学校和公司共同贡献的开源项目，我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户；同时，我们非常感谢 OpenSelfSup 的原开发者和贡献者。

我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## 引用
如果您发现此项目对您的研究有用，请考虑引用：

```bibtex
@misc{mmselfsup2021,
    title={OpenMMLab's Unsupervised Representation Learning Toolbox and Benchmark},
    author={MMSelfSup Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmselfsup}},
    year={2021}
}
```

## OpenMMLab 的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMLab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱与测试基准
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 检测工具箱与测试基准
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用3D目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱与测试基准
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 新一代生成模型工具箱
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=GJP18SjI)

<div align="center">
<img src="./resources/zhihu_qrcode.jpg" height="400"/>  <img src="./resources/qq_group_qrcode.jpg" height="400"/>
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
