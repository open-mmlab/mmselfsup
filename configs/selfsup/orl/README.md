# ORL

> [Unsupervised Object-Level Representation Learning
> from Scene Images
> ](https://arxiv.org/abs/2106.11952)

<!-- [ALGORITHM] -->

## Abstract

Contrastive self-supervised learning has largely narrowed the gap to supervised pre-training on ImageNet. However, its success highly relies on the object-centric priors of ImageNet, i.e., different augmented views of the same image correspond to the same object. Such a heavily curated constraint becomes immediately infeasible when pre-trained on more complex scene images with many objects. To overcome this limitation, we introduce Object-level Representation Learning (ORL), a new self-supervised learning framework towards scene images. Our key insight is to leverage image-level self-supervised pre-training as the prior to discover object-level semantic correspondence, thus realizing object-level representation learning from scene images. Extensive experiments on COCO show that ORL significantly improves the performance of self-supervised learning on scene images, even surpassing supervised ImageNet pre-training on several downstream tasks. Furthermore, ORL improves the downstream performance when more unlabeled scene images are available, demonstrating its great potential of harnessing unlabeled data in the wild. We hope our approach can motivate future research on more general-purpose unsupervised representation learning from scene data.

<div align="center">
<img src="https://github.com/Jiahao000/ORL/raw/2ad64f7389d20cb1d955792aabbe806a7097e6fb/highlights.png" width="90%" />
</div>

## Usage

ORL is mainly composed of three stages.
, e.g., BYOL. In Stage 2, we first use the pre-trained model to retrieve KNNs for each image in the embedding space to obtain image-level visually similar pairs. We then use unsupervised region proposal algorithms (e.g., selective search) to generate rough RoIs for each image pair. Afterwards, we reuse the pre-trained model to retrieve the top-ranked RoI pairs, i.e., correspondence. We find these pairs of RoIs are almost objects or object parts. In Stage 3, with the corresponding RoI pairs discovered across images, we finally perform object-level contrastive learning using the same architecture as Stage 1.

### Stage 1: Image-level pre-training

In Stage 1, ORL pre-trains an image-level contrastive learning model. In the end of pre-training, it will extract all features in the training set and retrieve KNNs for each image in the embedding space to obtain image-level visually similar pairs.

```shell
# Train with multiple GPUs
bash tools/dist_train.sh
configs/selfsup/orl/stage1/orl_resnet50_8xb64-coslr-800e_coco.py \
${GPUS} \
--work-dir work_dirs/selfsup/orl/stage1/orl_resnet50_8xb64-coslr-800e_coco/
```

or

```shell
# Train on cluster managed with slurm
GPUS_PER_NODE=${GPUS_PER_NODE} GPUS=${GPUS} CPUS_PER_TASK=${CPUS_PER_TASK} \
bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} \
configs/selfsup/orl/stage1/orl_resnet50_8xb64-coslr-800e_coco.py \
--work-dir work_dirs/selfsup/orl/stage1/orl_resnet50_8xb64-coslr-800e_coco/
```

The corresponding KNN image ids will be saved as a json file `train2017_knn_instance.json` under `../data/coco/meta/`.

### Stage 2: Correspondence discovery

- **RoI generation**

ORL applies selective search to generate region proposals for all images in the training set:

```shell
# Train with single GPU
bash tools/dist_selective_search_single_gpu.sh
configs/selfsup/orl/stage2/selective_search.py \
../data/coco/meta/train2017_selective_search_proposal.json \
--work-dir work_dirs/selfsup/orl/stage2/selective_search
```

or

```shell
# Train on cluster managed with slurm
GPUS_PER_NODE=${GPUS_PER_NODE} GPUS=1 CPUS_PER_TASK=${CPUS_PER_TASK} \
bash tools/slurm_selective_search_single_gpu.sh ${PARTITION} \
configs/selfsup/orl/stage2/selective_search.py \
../data/coco/meta/train2017_selective_search_proposal.json \
--work-dir work_dirs/selfsup/orl/stage2/selective_search
```

The script and config only support single-image single-gpu inference since different images can have different number of generated region proposals by selective search, which cannot be gathered if distributed in multiple gpus. You can also directly download [here](https://drive.google.com/drive/folders/1yYsyGiDjjVSOzIUkhxwO_NitUPLC-An_?usp=sharing) if you want to skip this step.

- **RoI pair retrieval**

ORL reuses the model pre-trained in stage 1 to retrieve the top-ranked RoI pairs, i.e., correspondence.

```shell
# Train with single GPU
bash tools/dist_generate_correspondence_single_gpu.sh
configs/selfsup/orl/stage2/correspondence.py \
work_dirs/selfsup/orl/stage1/orl_resnet50_8xb64-coslr-800e_coco/epoch_800.pth \
../data/coco/meta/train2017_10nn_instance.json \
../data/coco/meta/train2017_10nn_instance_correspondence.json \
--work-dir work_dirs/selfsup/orl/stage2/correspondence
```

or

```shell
# Train on cluster managed with slurm
GPUS_PER_NODE=${GPUS_PER_NODE} GPUS=1 CPUS_PER_TASK=${CPUS_PER_TASK} \
bash tools/slurm_selective_search_single_gpu.sh ${PARTITION} \
configs/selfsup/orl/stage2/correspondence.py \
work_dirs/selfsup/orl/stage1/orl_resnet50_8xb64-coslr-800e_coco/epoch_800.pth \
../data/coco/meta/train2017_10nn_instance.json \
../data/coco/meta/train2017_10nn_instance_correspondence.json \
--work-dir work_dirs/selfsup/orl/stage2/correspondence
```

The script and config also only support single-image single-gpu inference since different image pairs can have different number of generated inter-RoI pairs, which cannot be gathered if distributed in multiple gpus. It will save the final correspondence json file `train2017_knn_instance_correspondence.json` under `../data/coco/meta/`.

### Stage 3: Object-level pre-training

After obtaining the correspondence file in Stage 2, ORL then performs object-level pre-training:

```shell
# Train with multiple GPUs
bash tools/dist_train.sh
configs/selfsup/orl/stage3/orl_resnet50_8xb64-coslr-800e_coco.py \
${GPUS} \
--work-dir work_dirs/selfsup/orl/stage3/orl_resnet50_8xb64-coslr-800e_coco/
```

or

```shell
# Train on cluster managed with slurm
GPUS_PER_NODE=${GPUS_PER_NODE} GPUS=${GPUS} CPUS_PER_TASK=${CPUS_PER_TASK} \
bash tools/slurm_train.sh ${PARTITION} ${JOB_NAME} \
configs/selfsup/orl/stage3/orl_resnet50_8xb64-coslr-800e_coco.py \
--work-dir work_dirs/selfsup/orl/stage3/orl_resnet50_8xb64-coslr-800e_coco/
```

## Models and Benchmarks

Here, we report the Low-shot image classification results of the model, which is pre-trained on COCO train2017, we report mAP for each case across five runs and the details are below:

| Self-Supervised Config                                                                                                                                                                            | Best Layer | Weight                                                                                              | k=1   | k=2   | k=4   | k=8   | k=16  | k=32  | k=64  | k=96  |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------------------------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| [stage3/orl_resnet50_8xb64-coslr-800e_coco](https://github.com/zhaozh10/mmselfsup/blob/2b14f8b06e4ba2596e90f19e4bac0c13757d80f7/configs/selfsup/orl/stage3/orl_resnet50_8xb64-coslr-800e_coco.py) | feature5   | [Pre-trained](https://drive.google.com/drive/folders/1oWzNZpoN_SPc56Gr-l3AlgGSv8jG1izG?usp=sharing) | 42.25 | 51.81 | 63.46 | 72.16 | 77.86 | 81.17 | 83.73 | 84.59 |

## Citation

```bibtex
@inproceedings{xie2021unsupervised,
  title={Unsupervised Object-Level Representation Learning from Scene Images},
  author={Xie, Jiahao and Zhan, Xiaohang and Liu, Ziwei and Ong, Yew Soon and Loy, Chen Change},
  booktitle={NeurIPS},
  year={2021}
}
```
