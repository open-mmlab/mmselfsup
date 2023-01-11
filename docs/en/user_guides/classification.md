# Classification

- [Classification](#classification)
  - [VOC SVM / Low-shot SVM](#voc-svm--low-shot-svm)
  - [Linear Evaluation and Fine-tuning](#linear-evaluation-and-fine-tuning)
  - [ImageNet Semi-Supervised Classification](#imagenet-semi-supervised-classification)
  - [ImageNet Nearest-Neighbor Classification](#imagenet-nearest-neighbor-classification)

In MMSelfSup, we provide many benchmarks for classification, thus the models can be evaluated on different classification tasks. Here are comprehensive tutorials and examples to explain how to run all classification benchmarks with MMSelfSup.
We provide scripts in folder `tools/benchmarks/classification/`, which has 2 `.sh` files, 1 folder for VOC SVM related classification task and 1 folder for ImageNet nearest-neighbor classification task.

## VOC SVM / Low-shot SVM

To run these benchmarks, you should first prepare your VOC datasets. Please refer to [prepare_data.md](./2_dataset_prepare.md) for the details of data preparation.

To evaluate the pre-trained models, you can run the command below.

```shell
# distributed version
bash tools/benchmarks/classification/svm_voc07/dist_test_svm_pretrain.sh ${SELFSUP_CONFIG} ${GPUS} ${PRETRAIN} ${FEATURE_LIST}

# slurm version
bash tools/benchmarks/classification/svm_voc07/slurm_test_svm_pretrain.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${PRETRAIN} ${FEATURE_LIST}
```

Besides, if you want to evaluate the ckpt files saved by runner, you can run the command below.

```shell
# distributed version
bash tools/benchmarks/classification/svm_voc07/dist_test_svm_epoch.sh ${SELFSUP_CONFIG} ${EPOCH} ${FEATURE_LIST}

# slurm version
bash tools/benchmarks/classification/svm_voc07/slurm_test_svm_epoch.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${EPOCH} ${FEATURE_LIST}
```

**To test with ckpt, the code uses the epoch\_\*.pth file, there is no need to extract weights.**

Remarks:

- `${SELFSUP_CONFIG}` is the config file of the self-supervised experiment.
- `${FEATURE_LIST}` is a string to specify features from layer1 to layer5 to evaluate; e.g., if you want to evaluate layer5 only, then `FEATURE_LIST` is "feat5", if you want to evaluate all features, then `FEATURE_LIST` is "feat1 feat2 feat3 feat4 feat5" (separated by space). If left empty, the default `FEATURE_LIST` is "feat5".
- `${PRETRAIN}`: the pre-trained model file.
- if you want to change GPU numbers, you could add `GPUS_PER_NODE=4 GPUS=4` at the beginning of the command.
- `${EPOCH}` is the epoch number of the ckpt that you want to test

## Linear Evaluation and Fine-tuning

Linear evaluation and fine-tuning are two of the most general benchmarks. We provide config files and scripts to launch the training and testing
for Linear Evaluation and Fine-tuning. The supported datasets are **ImageNet**, **Places205** and **iNaturalist18**.

First, make sure you have installed [MIM](https://github.com/open-mmlab/mim), which is also a project of OpenMMLab.

```shell
pip install openmim
```

Besides, please refer to MMClassification for [installation](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/docs/en/install.md) and [data preparation](https://github.com/open-mmlab/mmclassification/blob/dev-1.x/docs/en/getting_started.md).

Then, run the command below.

```shell
# distributed version
bash tools/benchmarks/classification/mim_dist_train.sh ${CONFIG} ${PRETRAIN}

# slurm version
bash tools/benchmarks/classification/mim_slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${PRETRAIN}
```

Remarks:

- `${CONFIG}`: Use config files under `configs/benchmarks/classification/`. Specifically, `imagenet` (excluding `imagenet_*percent` folders), `places205` and `inaturalist2018`.
- `${PRETRAIN}`: the pre-trained model file.

Example:

```shell
bash ./tools/benchmarks/classification/mim_dist_train.sh \
configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-coslr-100e_in1k.py \
work_dir/pretrained_model.pth
```

If you want to test the well-trained model, please run the command below.

```shell
# distributed version
bash tools/benchmarks/classification/mim_dist_test.sh ${CONFIG} ${CHECKPOINT}

# slurm version
bash tools/benchmarks/classification//mim_slurm_test.sh ${PARTITION} ${CONFIG} ${CHECKPOINT}
```

Remarks:

- `${CHECKPOINT}`: The well-trained classification model that you want to test.

Example:

```shell
bash ./tools/benchmarks/mmsegmentation/mim_dist_test.sh \
configs/benchmarks/classification/imagenet/resnet50_linear-8xb32-coslr-100e_in1k.py \
work_dir/model.pth
```

## ImageNet Semi-Supervised Classification

To run ImageNet semi-supervised classification, we still use the same `.sh` script as Linear Evaluation and Fine-tuning to launch training.

Remarks:

- The default GPU number is 4.
- `${CONFIG}`: Use config files under `configs/benchmarks/classification/imagenet/`, named `imagenet_*percent` folders.
- `${PRETRAIN}`: the pre-trained model file.

## ImageNet Nearest-Neighbor Classification

```Note
Only support CNN-style backbones (like ResNet50).
```

To evaluate the pre-trained models using the nearest-neighbor benchmark, you can run the command below.

```shell
# distributed version
bash tools/benchmarks/classification/knn_imagenet/dist_test_knn.sh ${SELFSUP_CONFIG} ${PRETRAIN} [optional arguments]

# slurm version
bash tools/benchmarks/classification/knn_imagenet/slurm_test_knn.sh ${PARTITION} ${JOB_NAME} ${SELFSUP_CONFIG} ${CHECKPOINT} [optional arguments]
```

Remarks:

- `${SELFSUP_CONFIG}` is the config file of the self-supervised experiment.
- `${CHECKPOINT}`: the path of checkpoint model file.
- if you want to change GPU numbers, you could add `GPUS_PER_NODE=4 GPUS=4` at the beginning of the command.
- `[optional arguments]`: for optional arguments, you can refer to the [script](https://github.com/open-mmlab/mmselfsup/blob/1.x/tools/benchmarks/classification/knn_imagenet/test_knn.py)

An example of command

```shell
# distributed version
bash tools/benchmarks/classification/knn_imagenet/dist_test_knn.sh \
    configs/selfsup/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k.py \
    https://download.openmmlab.com/mmselfsup/1.x/barlowtwins/barlowtwins_resnet50_8xb256-coslr-300e_in1k/barlowtwins_resnet50_8xb256-coslr-300e_in1k_20220825-57307488.pth
```
