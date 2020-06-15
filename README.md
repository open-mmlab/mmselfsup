
# OpenSelfSup

## Introduction

The master branch works with **PyTorch 1.1** or higher.

`OpenSelfSup` is an open source unsupervised representation learning toolbox based on PyTorch.

### What does this repo do?

Below is the relations among Unsupervised Learning, Self-Supervised Learning and Representation Learning. This repo focuses on the shadow area, i.e., Unsupervised Representation Learning. Self-Supervised Representation Learning is the major branch of it. Since in many cases we do not distingush between Self-Supervised Representation Learning and Unsupervised Representation Learning strictly, we still name this repo as `OpenSelfSup`.

<img src="docs/relation.jpg" width="600"/>

### Major features

- **All methods in one repository**
  
|                                                                                                                                                       |  Support |
|-------------------------------------------------------------------------------------------------------------------------------------------------------|:--------:|
| [ImageNet](https://link.springer.com/article/10.1007/s11263-015-0816-y?sa_campaign=email/event/articleAuthor/onlineFirst#)                            |     ✓    |
| [Relative-Loc](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf)      |     ✓    |
| [Rotation-Pred](https://arxiv.org/abs/1803.07728)                                                                                                     |     ✓    |
| [DeepCluster](https://arxiv.org/abs/1807.05520)                                                                                                       |     ✓    |
| [ODC](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf) |     ✓    |
| [NIPD](https://arxiv.org/abs/1805.01978)                                                                                                              |     ✓    |
| [MoCo](https://arxiv.org/abs/1911.05722)                                                                                                              |     ✓    |
| [MoCo v2](https://arxiv.org/abs/2003.04297)                                                                                                           |     ✓    |
| [SimCLR](https://arxiv.org/abs/2002.05709)                                                                                                            |     ✓    |
| [PIRL](http://openaccess.thecvf.com/content_CVPR_2020/papers/Misra_Self-Supervised_Learning_of_Pretext-Invariant_Representations_CVPR_2020_paper.pdf) | progress |

- **Flexibility & Extensibility**

`OpenSelfSup` follows a similar code architecture of MMDetection while is even more flexible than MMDetection, since OpenSelfSup integrates various self-supervised tasks including classification, joint clustering and feature learning, contrastive learning, tasks with a memory bank, etc.

For existing methods in this repo, you only need to modify config files to adjust hyper-parameters. It is also simple to design your own methods, please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md).

- **Efficiency**

  All methods support multi-machine multi-gpu distributed training.

- **Standardized Benchmarks**

  We standardize the benchmarks including logistic regression, SVM / Low-shot SVM from linearly probed features, semi-supervised classification, and object detection. Below are the setting of these benchmarks.

| Benchmarks                       | Setting                                                                                                                                                                     | Difference                                      |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|
| ImageNet Linear Classification   | [goyal2019scaling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) | Total 90 epochs, decay at [30, 60].             |
| Places205 Linear Classification  | [goyal2019scaling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) | Total 90 epochs, decay at [30, 60].             |
| ImageNet Semi-Sup Classification |
| PASCAL VOC07 SVM                 | [goyal2019scaling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) | Costs="1.0,10.0,100.0" to save evaluation time. |
| PASCAL VOC07 Low-shot SVM        | [goyal2019scaling](http://openaccess.thecvf.com/content_ICCV_2019/papers/Goyal_Scaling_and_Benchmarking_Self-Supervised_Visual_Representation_Learning_ICCV_2019_paper.pdf) | Costs="1.0,10.0,100.0" to save evaluation time. |
| PASCAL VOC07+12 Object Detection | [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)                      |                                                 |
| COCO17 Object Detection          | [MoCo](http://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)                      |                                                 |

## Change Log

Please refer to [CHANGELOG.md](docs/CHANGELOG.md) for details and release history.

[2020-06-16] `OpenSelfSup` v0.1.0 is released.

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for installation and dataset preparation.

## Get Started

Please see [GETTING_STARTED.md](docs/GETTING_STARTED.md) for the basic usage of OpenSelfSup.

## Benchmark and Model Zoo

Please refer to [MODEL_ZOO.md](docs/MODEL_ZOO.md) for for a comprehensive set of pre-trained models and benchmarks.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{openselfsup,
  title   = {{OpenSelfSup}: Open MMLab Self-Supervised Learning Toolbox and Benchmark},
  author  = {Xiaohang Zhan, Jiahao Xie, Ziwei Liu, Dahua Lin, Chen Change Loy},
  howpublished = {\url{https://github.com/open-mmlab/openselfsup}},
  year = {2020}
}
```

## Acknowledgement

- This repo borrows the architecture design and part of the code from [MMDetection](https://github.com/open-mmlab/mmdetection).
- The implementation of MoCo and the detection benchmark borrow the code from [moco](https://github.com/facebookresearch/moco).
- The SVM benchmark borrows the code from [
fair_self_supervision_benchmark](https://github.com/facebookresearch/fair_self_supervision_benchmark).
- `openselfsup/third_party/clustering.py` is borrowed from [deepcluster](https://github.com/facebookresearch/deepcluster/blob/master/clustering.py).

## Contact

This repo is currently maintained by Xiaohang Zhan ([@XiaohangZhan](http://github.com/XiaohangZhan)).
