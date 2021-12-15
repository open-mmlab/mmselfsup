# NPID

## Unsupervised Feature Learning via Non-Parametric Instance Discrimination

<!-- [ABSTRACT] -->

Neural net classifiers trained on data with annotated class labels can also capture apparent visual similarity among categories without being directed to do so. We study whether this observation can be extended beyond the conventional domain of supervised learning: Can we learn a good feature representation that captures apparent similar- ity among instances, instead of classes, by merely asking the feature to be discriminative of individual instances?

We formulate this intuition as a non-parametric classification problem at the instance-level, and use noise-contrastive estimation to tackle the computational challenges imposed by the large number of instance classes. Our experimental results demonstrate that, under unsupervised learning settings, our method surpasses the state-of-the-art on ImageNet classification by a large margin.

Our method is also remarkable for consistently improving test performance with more training data and better network architectures. By fine-tuning the learned feature, we further obtain competitive results for semi-supervised learning and object detection tasks. Our non-parametric model is highly compact: With 128 features per image, our method requires only 600MB storage for a million images, enabling fast nearest neighbour retrieval at the run time.

<!-- [IMAGE] -->
<div align="center">
<img  />
</div>

## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{wu2018unsupervised,
  title={Unsupervised feature learning via non-parametric instance discrimination},
  author={Wu, Zhirong and Xiong, Yuanjun and Yu, Stella X and Lin, Dahua},
  booktitle={CVPR},
  year={2018}
}
```

## Models and Benchmarks

[Back to model_zoo.md](../../../docs/model_zoo.md)

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models were trained on ImageNet1k dataset.


### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Model     | Config                                                                | Best Layer | SVM | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 | k=64 | k=96 |
| --------- | --------------------------------------------------------------------- | ---------- | --- | --- | --- | --- | --- | ---- | ---- | ---- | ---- |
| [model]() | [resnet50_8xb32-steplr-200e](npid_resnet50_8xb32-steplr-200e_in1k.py) |            |     |     |     |     |     |      |      |      |      |

### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_8xb32-steplr-90e.py](../../benchmarks/classification/imagenet/resnet50_mhead_8xb32-steplr-90e_in1k.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [file name]() for details of config.

| Model     | Config                                                                | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| --------- | --------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | ------- |
| [model]() | [resnet50_8xb32-steplr-200e](npid_resnet50_8xb32-steplr-200e_in1k.py) |          |          |          |          |          |         |

### iNaturalist2018 Linear Evaluation

Please refer to [resnet50_mhead_8xb32-steplr-84e_inat18.py](../../benchmarks/classification/inaturalist2018/resnet50_mhead_8xb32-steplr-84e_inat18.py) and [file name]() for details of config.

| Model     | Config                                                                | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| --------- | --------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | ------- |
| [model]() | [resnet50_8xb32-steplr-200e](npid_resnet50_8xb32-steplr-200e_in1k.py) |          |          |          |          |          |         |

### Places205 Linear Evaluation

Please refer to [resnet50_mhead_8xb32-steplr-28e_places205.py](../../benchmarks/classification/inaturalist2018/resnet50_mhead_8xb32-steplr-28e_places205.py) and [file name]() for details of config.

| Model     | Config                                                                | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| --------- | --------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | ------- |
| [model]() | [resnet50_8xb32-steplr-200e](npid_resnet50_8xb32-steplr-200e_in1k.py) |          |          |          |          |          |         |

#### Semi-Supervised Classification

- In this benchmark, the necks or heads are removed and only the backbone CNN is evaluated by appending a linear classification head. All parameters are fine-tuned.
- When training with 1% ImageNet, we find hyper-parameters especially the learning rate greatly influence the performance. Hence, we prepare a list of settings with the base learning rate from `{0.001, 0.01, 0.1}` and the learning rate multiplier for the head from `{1, 10, 100}`. We choose the best performing setting for each method. The setting of parameters are indicated in the file name. The learning rate is indicated like `1e-1`, `1e-2`, `1e-3` and the learning rate multiplier is indicated like `head1`, `head10`, `head100`.
- Please use --deterministic in this benchmark.

Please refer to the directories `configs/benchmarks/classification/imagenet/imagenet_1percent/` of 1% data and `configs/benchmarks/classification/imagenet/imagenet_10percent/` 10% data for details.

| Model     | Pretrain Config                                                       | Fine-tuned Config | Top-1 (%) | Top-5 (%) |
| --------- | --------------------------------------------------------------------- | ----------------- | --------- | --------- |
| [model]() | [resnet50_8xb32-steplr-200e](npid_resnet50_8xb32-steplr-200e_in1k.py) |                   |           |           |

### Detection

The detection benchmarks includes 2 downstream task datasets, **Pascal VOC 2007 + 2012** and **COCO2017**. This benchmark follows the evluation protocols set up by MoCo.

#### Pascal VOC 2007 + 2012

Please refer to [faster_rcnn_r50_c4_mstrain_24k.py](../../benchmarks/mmdetection/voc0712/faster_rcnn_r50_c4_mstrain_24k.py) for details of config.

| Model     | Config                                                                | mAP | AP50 |
| --------- | --------------------------------------------------------------------- | --- | ---- |
| [model]() | [resnet50_8xb32-steplr-200e](npid_resnet50_8xb32-steplr-200e_in1k.py) |     |      |

#### COCO2017

Please refer to [mask_rcnn_r50_fpn_mstrain_1x.py](../../benchmarks/mmdetection/coco/mask_rcnn_r50_fpn_mstrain_1x.py) for details of config.

| Model     | Config                                                                | mAP(Box) | AP50(Box) | AP75(Box) | mAP(Mask) | AP50(Mask) | AP75(Mask) |
| --------- | --------------------------------------------------------------------- | -------- | --------- | --------- | --------- | ---------- | ---------- |
| [model]() | [resnet50_8xb32-steplr-200e](npid_resnet50_8xb32-steplr-200e_in1k.py) |          |           |           |           |            |            |

### Segmentation

The segmentation benchmarks includes 2 downstream task datasets, **Cityscapes** and **Pascal VOC 2012 + Aug**. It follows the evluation protocols set up by MMSegmentation.

#### Pascal VOC 2012 + Aug

Please refer to [file]() for details of config.

| Model     | Config                                                                | mIOU |
| --------- | --------------------------------------------------------------------- | ---- |
| [model]() | [resnet50_8xb32-steplr-200e](npid_resnet50_8xb32-steplr-200e_in1k.py) |      |


#### Cityscapes

Please refer to [file]() for details of config.

| Model     | Config                                                                | mIOU |
| --------- | --------------------------------------------------------------------- | ---- |
| [model]() | [resnet50_8xb32-steplr-200e](npid_resnet50_8xb32-steplr-200e_in1k.py) |      |
