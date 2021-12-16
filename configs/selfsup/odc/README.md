# ODC

## Online Deep Clustering for Unsupervised Representation Learning

<!-- [ABSTRACT] -->

Joint clustering and feature learning methods have shown remarkable performance in unsupervised representation learning. However, the training schedule alternating between feature clustering and network parameters update leads to unstable learning of visual representations. To overcome this challenge, we propose Online Deep Clustering (ODC) that performs clustering and network update simultaneously rather than alternatingly. Our key insight is that the cluster centroids should evolve steadily in keeping the classifier stably updated. Specifically, we design and maintain two dynamic memory modules, i.e., samples memory to store samples’ labels and features, and centroids memory for centroids evolution. We break down the abrupt global clustering into steady memory update and batch-wise label re-assignment. The process is integrated into network update iterations. In this way, labels and the network evolve shoulder-to-shoulder rather than alternatingly. Extensive experiments demonstrate that ODC stabilizes the training process and boosts the performance effectively.

<!-- [IMAGE] -->
<div align="center">
<img  />
</div>

## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{zhan2020online,
  title={Online deep clustering for unsupervised representation learning},
  author={Zhan, Xiaohang and Xie, Jiahao and Liu, Ziwei and Ong, Yew-Soon and Loy, Chen Change},
  booktitle={CVPR},
  year={2020}
}
```

## Models and Benchmarks

**Back to [model_zoo.md](../../../docs/en/model_zoo.md) to download models.**

In this page, we provide benchmarks as much as possible to evaluate our pre-trained models. If not mentioned, all models were trained on ImageNet1k dataset.

### Classification

The classification benchmarks includes 4 downstream task datasets, **VOC**, **ImageNet**,  **iNaturalist2018** and **Places205**. If not specified, the results are  Top-1 (%).

#### VOC SVM / Low-shot SVM

The **Best Layer** indicates that the best results are obtained from which layers feature map. For example, if the **Best Layer** is **feature3**, its best result is obtained from the second stage of ResNet (1 for stem layer, 2-5 for 4 stage layers).

Besides, k=1 to 96 indicates the hyper-parameter of Low-shot SVM.

| Self-Supervised Config                                               | Best Layer | SVM   | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| -------------------------------------------------------------------- | ---------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [resnet50_8xb64-steplr-440e](odc_resnet50_8xb64-steplr-440e_in1k.py) | feature5   | 78.42 | 32.42 | 40.27 | 49.95 | 59.96 | 65.71 | 69.99 | 73.64 | 75.13 |

#### ImageNet Linear Evaluation

The **Feature1 - Feature5** don't have the GlobalAveragePooling, the feature map is pooled to the specific dimensions and then follows a Linear layer to do the classification. Please refer to [resnet50_mhead_8xb32-steplr-90e.py](../../benchmarks/classification/imagenet/resnet50_mhead_8xb32-steplr-90e_in1k.py) for details of config.

The **AvgPool** result is obtained from Linear Evaluation with GlobalAveragePooling. Please refer to [resnet50_8xb32-steplr-100e_in1k](../../benchmarks/classification/imagenet/resnet50_8xb32-steplr-100e_in1k.py) for details of config.

| Self-Supervised Config                                               | Feature1 | Feature2 | Feature3 | Feature4 | Feature5 | AvgPool |
| -------------------------------------------------------------------- | -------- | -------- | -------- | -------- | -------- | ------- |
| [resnet50_8xb64-steplr-440e](odc_resnet50_8xb64-steplr-440e_in1k.py) | 14.76    | 31.82    | 42.44    | 55.76    | 57.70    | 53.42   |
