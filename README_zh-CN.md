<div align="center">
  <img src="./resources/mmselfsup_logo.png" width="500"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmselfsup)](https://pypi.org/project/mmselfsup)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmselfsup.readthedocs.io/en/dev-1.x/)
[![badge](https://github.com/open-mmlab/mmselfsup/workflows/build/badge.svg)](https://github.com/open-mmlab/mmselfsup/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmselfsup/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmselfsup)
[![license](https://img.shields.io/github/license/open-mmlab/mmselfsup.svg)](https://github.com/open-mmlab/mmselfsup/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmselfsup.svg)](https://github.com/open-mmlab/mmselfsup/issues)

[📘使用文档](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/) |
[🛠️安装教程](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/get_started.html) |
[👀模型库](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/model_zoo.html) |
[🆕更新日志](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/notes/changelog.html) |
[🤔报告问题](https://github.com/open-mmlab/mmselfsup/issues/new/choose)

</div>

<div align="center">

[English](README.md) | 简体中文

</div>

## 介绍

MMSelfSup 是一个基于 PyTorch 实现的开源自监督表征学习工具箱，是 [OpenMMLab](https://openmmlab.com/) 项目成员之一。

主分支代码支持 **PyTorch 1.6** 及以上的版本。

### 主要特性

- **多方法集成**

  MMSelfSup 提供了多种前沿的自监督学习算法，大部分的自监督预训练学习都设置相同，以在基准中获得更加公平的比较。

- **模块化设计**

  MMSelfSup 遵照 OpenMMLab 项目一贯的设计理念，进行模块化设计，便于用户自定义实现自己的算法。

- **标准化的性能评测**

  MMSelfSup 拥有丰富的基准进行评估和测试，包括线性评估, 线性特征的 SVM / Low-shot SVM, 半监督分类, 目标检测和语义分割。

- **兼容性**

  兼容 OpenMMLab 各大算法库，拥有丰富的下游评测任务和预训练模型的应用。

## 更新

**v1.0.0rc3** 版本已经在 2022.11.01 发布。

- 支持 MaskFeat
- 更新配置文件和文档

**v1.0.0rc2** 版本已经在 2022.10.12 发布。

- `MAE`, `SimMIM`, `MoCoV3` 算法模型的全量支持（支持多种模型尺寸和预训练轮次）。
- 更新文档以及教程

请参考 [更新日志](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/notes/changelog.html) 获取更多细节和历史版本信息。

MMSelfSup 1.x 和 0.x 的不同点写在 [迁移文档](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/migration.html) 中。

## 安装

MMSelfSup 依赖 [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv), [MMEngine](https://github.com/open-mmlab/mmengine) 和 [MMClassification](https://github.com/open-mmlab/mmclassification)。

请参考 [安装文档](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/get_started.html) 获取更详细的安装指南。

## 快速入门

我们针对基础使用和进阶用法提供了 [用户指引](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/user_guides/index.html)：

Pretrain

- [Config](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/user_guides/1_config.html)
- [Prepare Dataset](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/user_guides/2_dataset_prepare.html)
- [Pretrain with Existing Models](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/user_guides/3_pretrain.html)

Downetream Tasks

- [Classification](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/user_guides/classification.html)
- [Detection](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/user_guides/detection.html)
- [Segmentation](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/user_guides/segmentation.html)

Useful Tools

- [Visualization](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/user_guides/visualization.html)
- [Analysis Tools](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/user_guides/analysis_tools.html)

我们也提供了 [进阶指引](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/advanced_guides/index.html) 和 [Colab 教程](https://github.com/open-mmlab/mmselfsup/blob/master/demo/mmselfsup_colab_tutorial.ipynb)。

如果遇到问题，请参考 [常见问题解答](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/notes/faq.html)。

## 模型库

请参考 [模型库](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/model_zoo.html) 查看我们更加全面的模型基准结果。

目前已支持的算法:

- [x] [Relative Location (ICCV'2015)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/relavive_loc)
- [x] [Rotation Prediction (ICLR'2018)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/rotation_pred)
- [x] [DeepCluster (ECCV'2018)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/deepcluster)
- [x] [NPID (CVPR'2018)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/npid)
- [x] [ODC (CVPR'2020)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/odc)
- [x] [MoCo v1 (CVPR'2020)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/mocov1)
- [x] [SimCLR (ICML'2020)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/simclr)
- [x] [MoCo v2 (ArXiv'2020)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/byol)
- [x] [BYOL (NeurIPS'2020)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/mocov2)
- [x] [SwAV (NeurIPS'2020)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/swav)
- [x] [DenseCL (CVPR'2021)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/densecl)
- [x] [SimSiam (CVPR'2021)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/simsiam)
- [x] [Barlow Twins (ICML'2021)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/barlowtwins)
- [x] [MoCo v3 (ICCV'2021)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/mocov3)
- [x] [MAE (CVPR'2022)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/mae)
- [x] [SimMIM (CVPR'2022)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/simmim)
- [x] [MaskFeat (CVPR'2022)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/maskfeat)
- [x] [CAE (ArXiv'2022)](https://github.com/open-mmlab/mmselfsup/tree/dev-1.x/configs/selfsup/cae)

更多的算法实现已经在我们的计划中。

## 基准测试

| 基准测试方法                                       | 参考设置                                                                                                                                                             |
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

## 参与贡献

我们非常欢迎任何有助于提升 MMSelfSup 的贡献，请参考 [贡献指南](https://mmselfsup.readthedocs.io/zh_CN/dev-1.x/notes/contribution_guide.html) 来了解如何参与贡献。

## 致谢

MMSelfSup 是一款由不同学校和公司共同贡献的开源项目，我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户；同时，我们非常感谢 OpenSelfSup 的原开发者和贡献者。

我们希望该工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现现有算法并开发自己的新模型，从而不断为开源社区提供贡献。

## 引用

如果您发现此项目对您的研究有用，请考虑引用：

```bibtex
@misc{mmselfsup2021,
    title={{MMSelfSup}: OpenMMLab Self-Supervised Learning Toolbox and Benchmark},
    author={MMSelfSup Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmselfsup}},
    year={2021}
}
```

## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## OpenMMLab 的其他项目

- [MMEngine](https://github.com/open-mmlab/mmengine): OpenMMLab 深度学习模型训练基础库
- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MMEval](https://github.com/open-mmlab/mmeval): 统一开放的跨框架算法评测库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO 系列工具箱和基准测试
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具箱
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=GJP18SjI)，添加OpenMMLab 官方小助手微信，加入 MMSelfSup 微信社区。

<div align="center">
<img src="./resources/zhihu_qrcode.jpg" height="400"/>  <img src="./resources/qq_group_qrcode.jpg" height="400"/> <img src="./resources/xiaozhushou_weixin_qrcode.jpeg" height="400"/>
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
