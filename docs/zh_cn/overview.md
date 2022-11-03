# 总体介绍

- [总体介绍](#总体介绍)
  - [自监督学习](#自监督学习)
  - [MMSelfSup的架构设计](#mmselfsup的架构设计)
  - [MMSelfSup的上手路线图](#mmselfsup的上手路线图)
    - [使用MMSelfSup玩转自监督实验](#使用mmselfsup玩转自监督实验)
    - [基于MMSelfSup学习自监督算法](#基于mmselfsup学习自监督算法)

在本文档中，我们将对[MMSelfSup](https://github.com/open-mmlab/mmselfsup)进行整体介绍。
我们首先会对自监督学习(Self-supervised Learning)的基本概念进行回顾，然后简单介绍整个MMSelfSup的基本架构。最后，我们将给出MMSelfSup的上手路线图，帮助大家更快的使用MMSelfSup助力自己的科学研究和项目实践。

## 自监督学习

自监督学习(Self-supervised learning, SSL)是一种极具潜力的学习范式，它旨在使用海量的无标注数据来进行表征学习。在SSL中，我们通过构造合理的预训练任务（可自动生成标注，即自监督）来进行模型的训练，学习到一个具有强大建模能力的预训练模型。基于自监督学习获得的训练模型，我们可以提升各类下游视觉任务(图像分类，物体检测，语义分割等)的性能。

过去几年里，自监督学习的研究获得了快速发展，整个学术社区涌现了一大批优秀的自监督学习算法。我们旨在将MMSelfSup打造成为一个功能强大，方便易用的开源算法库，助力学术研究和工程实践。接下来，我们将介绍MMSelfSup的架构设计。

## MMSelfSup的架构设计

与其他OpenMMLab的项目类似，MMSelfSup基于模块化设计的准则，整体的架构设计如下图所示：

<div align="center">
  <img src="https://user-images.githubusercontent.com/36138628/199443908-e7fd3670-108b-46eb-b200-d76f25e5621b.jpg" width="500"/>
</div>

- **Datasets** 支持各类数据集，同时提供丰富的数据增强策略。
- **Algorithms** 包括了多个经典的自监督算法，同时提供易用的用户接口。
- **Tools** 包括了自监督学习常用的训练和分析工具。
- **Benchmarks** 提供了使用自监督学习获得的预训练模型进行多种下游任务((图像分类，物体检测，语义分割等)的示例。

## MMSelfSup的上手路线图

为了帮助用户更快的上手MMSelfSup，我们推荐参考如下的路线图进行相关学习。

### 使用MMSelfSup玩转自监督实验

通常自监督学习算法被认为是一种适用不同模型架构的预训练算法，因此自监督学习应用通常会包括**预训练阶段**和**下游任务迁移学习阶段**。

- 如果你想尝试MMSelfSup提供的各类自监督学习算法，我们推荐你优先参考[Get Started](./get_started.md) 来进行**环境配置**。

- 对于**预训练阶段**，我们推荐你参考[Pre-train](user_guides/3_pretrain.md) 来尝试各类预训练算法，获得预训练模型。

- 对于**下游任务迁移学习阶段**，我们推荐你参考[Benchmark](https://mmselfsup.readthedocs.io/en/dev-1.x/user_guides/#downstream-tasks) 当中提供的示例，来使用预训练模型来尝试各种下游任务。

- 除此之外，我们也提供了多种分析工具和可视化工具[Useful Tools](https://mmselfsup.readthedocs.io/en/dev-1.x/user_guides/#useful-tools)来帮助用户更方便地对算法进行诊断和分析。

### 基于MMSelfSup学习自监督算法

如果你是对SSL不太了解的新同学，我们推荐你将[Model Zoo](model_zoo.md)作为一个参考，学习我们已经支持的一些自监督学习的代表性工作。
