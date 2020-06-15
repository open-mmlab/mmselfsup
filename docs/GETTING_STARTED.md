# Getting Started

This page provides basic tutorials about the usage of OpenSelfSup.
For installation instructions, please see [INSTALL.md](INSTALL.md).

## Train existing methods

**Note**: The default learning rate in config files is for 8 GPUs (except for those under `configs/linear_classification` that use 1 GPU). If using differnt number GPUs, the total batch size will change in proportion, you have to scale the learning rate following `new_lr = old_lr * new_ngpus / old_ngpus`. We recommend to use `tools/dist_train.sh` even with 1 gpu, since some methods do not support non-distributed training.

### Train with single/multiple GPUs
```shell
# checkpoints and logs are saved in the same sub-directory as the config file under `work_dirs/` by default.
bash tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

An example:
```shell
bash tools/dist_train.sh configs/selfsup/odc/r50_v1.py 8
```

Optional arguments are:
- `--work_dir ${WORK_DIR}`: Override the default working directory.
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--pretrained ${PRETRAIN_WEIGHTS}`: Load pretrained weights for the backbone.

Alternatively, if you run OpenSelfSup on a cluster managed with [slurm](https://slurm.schedmd.com/):
```shell
SRUN_ARGS="${SRUN_ARGS}" bash tools/srun_train.sh ${PARTITION} ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

An example:
```shell
SRUN_ARGS="-w xx.xx.xx.xx" bash tools/srun_train.sh Dummy configs/selfsup/odc/r50_v1.py 8
```

### Launch multiple jobs on a single machine

If you launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
you need to specify different ports (29500 by default) for each job to avoid communication conflict.

If you use `dist_train.sh` to launch training jobs:
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 bash tools/dist_train.sh ${CONFIG_FILE} 4
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 bash tools/dist_train.sh ${CONFIG_FILE} 4
```

If you use launch training jobs with slurm:
```shell
GPUS_PER_NODE=4 bash tools/srun_train.sh ${PARTITION} ${CONFIG_FILE} 4 --port 29500
GPUS_PER_NODE=4 bash tools/srun_train.sh ${PARTITION} ${CONFIG_FILE} 4 --port 29501
```

## Benchmarks

We provide several standard benchmarks to evaluate representation learning.

### VOC07 Linear SVM & Low-shot Linear SVM

```shell
bash benchmarks/dist_test_svm.sh ${CONFIG_FILE} ${EPOCH} ${FEAT_LIST} ${GPU_NUM}
```
Augments:
- `${FEAT_LIST}` is a string to specify features from layer1 to layer5 to evaluate; e.g., if you want to evaluate layer5 only, then `FEAT_LIST` is `feat5`, if you want to evaluate all features, then then `FEAT_LIST` is `feat1 feat2 feat3 feat4 feat5` (separated by space).
- `$GPU_NUM` is the number of GPUs to extract features.

### ImageNet / Places205 Linear Classification

```shell
bash benchmarks/dist_test_cls.sh ${CONFIG_FILE} ${EPOCH} ${DATASET} [optional arguments]
```
Augments:
- `${DATASET}` in `['imagenet', 'places205']`.
- Optional arguments include `--resume_from ${CHECKPOINT_FILE}` that resume from a previous checkpoint file.

### VOC07+12 / COCO17 Object Detection

1. First, extract backbone weights:

    ```shell
    python tools/extract_backbone_weights.py ${CHECKPOINT} --save-path ${WEIGHT_FILE}
    ```
    Arguments:
    - `CHECKPOINTS`: the checkpoint file of a selfsup method named as `epoch_*.pth`.
    - `WEIGHT_FILE`: the output backbone weights file, e.g., `odc_v1.pth`.
    
2. Next, run detection. For more details to setup the environments for detection, please refer [here](benchmarks/detection/README.md).
```shell
conda activate detectron2
cd benchmarks/detection
python convert-pretrain-to-detectron2.py ${WEIGHT_FILE} ${OUTPUT_FILE} # must use .pkl as the output extension.
bash run.sh ${DET_CFG} ${OUTPUT_FILE}
```
Arguments:
- `DET_CFG`: the detectron2 config file, usually we use `configs/pascal_voc_R_50_C4_24k_moco.yaml`.
- `OUTPUT_FILE`: converted backbone weights file, e.g., `odc_v1.pkl`.

**Note**:
- This benchmark must use 8 GPUs as the default setting from MoCo.
- Please report the mean of 5 trials in your offical paper, according to MoCo.
- DeepCluster that uses Sobel layer is not supported by detectron2.

### Publish a model

1. Extract the backbone weights as mentioned before. You don't have to extract it again if you've already done it in the benchmark step.

```shell
python tools/extract_backbone_weights.py ${CHECKPOINT} --save-path ${WEIGHT_FILE}
```

2. Compute the hash of the weight file and append the hash id to the filename.

```shell
python tools/publish_model.py ${WEIGHT_FILE}
```

## How-to

### Use a new dataset

1. Write a data source file under `openselfsup/datasets/data_sources/`. You may refer to the existing ones.

2. Create new config files for your experiments.

### Design your own methods

#### What you need to do

    1. Create a dataset file under `openselfsup/datasets/` (better using existing ones);
    2. Create a model file under `openselfsup/models/`. The model typically contains:
      i) backbone (required): images to deep features from differet depth of layers.
      ii) neck (optional): deep features to compact feature vectors.
      iii) head (optional): define loss functions.
      iv) memory_bank (optional): define memory banks.
    3. Create a config file under `configs/` and setup the configs;
    4. Create a hook file under `openselfsup/hooks/` if your method requires additional operations before run, every several iterations, every several epoch, or after run.
    
You may refer to existing modules under respective folders.

#### Features may facilitate your implementation

* Decoupled data source and dataset.

Since dataset is correlated to a specific task while data source is general, we decouple data source and dataset in OpenSelfSup.

```python
data = dict(
    train=dict(type='ContrastiveDataset',
               data_source=dict(type='ImageNet', list_file='xx', root='xx'),
               pipeline=train_pipeline),
    val=dict(...),
)
```

* Configure data augmentations in the config file.

The augmentations are the same as `torchvision.transforms`. `torchvision.transforms.RandomAppy` corresponds to `RandomAppliedTrans`. `Lighting` and `GaussianBlur` is additionally implemented.

```python
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomAppliedTrans',
        transforms=[
            dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, kernel_size=23)],
        p=0.5),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]
```

* Parameter-wise optimization parameters.

You may specify optimization paramters including lr, momentum and weight_decay for a certain group of paramters in the config file with `paramwise_options`. `paramwise_options` is a dict whose key is regular expressions and value is options. Options include 6 fields: lr, lr_mult, momentum, momentum_mult, weight_decay, weight_decay_mult.

```python
paramwise_options = {
    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay_mult=0.1),
    '\Ahead.': dict(lr_mult=10, momentum=0)}
optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
                     weight_decay=0.0001,
                     paramwise_options=paramwise_options)
```

* Configure custom hooks in the config file.

The hooks will be called in order. For hook design, please refer to [odc_hook.py](openselfsup/hooks/odc_hook.py) as an example.

```python
custom_hooks = [
    dict(type='DeepClusterHook', **kwargs1),
    dict(type='ODCHook', **kwargs2),
]
```
