# Getting Started

This page provides basic tutorials about the usage of OpenSelfSup.
For installation instructions, please see [INSTALL.md](INSTALL.md).

## Train existing methods

**Note**: The default learning rate in config files is for 8 GPUs. If using differnt number GPUs, the total batch size will change in proportion, you have to scale the learning rate following `new_lr = old_lr * new_ngpus / old_ngpus`. We recommend to use `tools/dist_train.sh` even with 1 gpu, since some methods do not support non-distributed training.

### Train with single/multiple GPUs

```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]
```
Optional arguments are:
- `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--pretrained ${PRETRAIN_WEIGHTS}`: Load pretrained weights for the backbone.
- `--deterministic`: Switch on "deterministic" mode which slows down training but the results are reproducible.

An example:
```shell
# checkpoints and logs saved in WORK_DIR=work_dirs/selfsup/odc/r50_v1/
bash tools/dist_train.sh configs/selfsup/odc/r50_v1.py 8
```
**Note**: During training, checkpoints and logs are saved in the same folder structure as the config file under `work_dirs/`. Custom work directory is not recommended since evaluation scripts infer work directories from the config file name. If you want to save your weights somewhere else, please use symlink, for example:

```shell
ln -s /DATA/xhzhan/openselfsup_workdirs ${OPENSELFSUP}/work_dirs
```

Alternatively, if you run OpenSelfSup on a cluster managed with [slurm](https://slurm.schedmd.com/):
```shell
SRUN_ARGS="${SRUN_ARGS}" bash tools/srun_train.sh ${PARTITION} ${CONFIG_FILE} ${GPUS} [optional arguments]
```

An example:
```shell
SRUN_ARGS="-w xx.xx.xx.xx" bash tools/srun_train.sh Dummy configs/selfsup/odc/r50_v1.py 8 --resume_from work_dirs/selfsup/odc/r50_v1/epoch_100.pth
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

### What if I do not have so many GPUs?

Assuming that you only have 1 GPU that can contain 64 images in a batch, while you expect the batch size to be 256, you may add the following line into your config file. It performs network update every 4 iterations. In this way, the equivalent batch size is 256. Of course, it is about 4x slower than using 4 GPUs. Note that the workaround is not applicable for methods like SimCLR which require intra-batch communication.

```python
optimizer_config = dict(update_interval=4)
```

### Mixed Precision Training (Optional)
We use [Apex](https://github.com/NVIDIA/apex) to implement Mixed Precision Training. 
If you want to use Mixed Precision Training, you can add below in the config file.
```python
use_fp16 = True
optimizer_config = dict(use_fp16=use_fp16)
```
An example:
```python
bash tools/dist_train.sh configs/selfsup/moco/r50_v1_fp16.py 8
```


## Benchmarks

We provide several standard benchmarks to evaluate representation learning. The config files or scripts for evaluation mentioned below are NOT recommended to be changed if you want to use this repo in your publications. We hope that all methods are under a fair comparison.

### VOC07 Linear SVM & Low-shot Linear SVM

```shell
# test by epoch (only applicable to experiments trained with OpenSelfSup)
bash benchmarks/dist_test_svm_epoch.sh ${CONFIG_FILE} ${EPOCH} ${FEAT_LIST} ${GPUS}
# test a pretrained model (applicable to any pre-trained models)
bash benchmarks/dist_test_svm_pretrain.sh ${CONFIG_FILE} ${PRETRAIN} ${FEAT_LIST} ${GPUS}
```
Augments:
- `${CONFIG_FILE}` the config file of the self-supervised experiment.
- `${FEAT_LIST}` is a string to specify features from layer1 to layer5 to evaluate; e.g., if you want to evaluate layer5 only, then `FEAT_LIST` is `"feat5"`, if you want to evaluate all features, then then `FEAT_LIST` is `"feat1 feat2 feat3 feat4 feat5"` (separated by space). If left empty, the default `FEAT_LIST` is `"feat5"`.
- `$GPUS` is the number of GPUs to extract features.

Working directories:
The features, logs and intermediate files generated are saved in `$SVM_WORK_DIR/` as follows:
- `dist_test_svm_epoch.sh`: `SVM_WORK_DIR=$WORK_DIR/` (The same as that mentioned in `Train with single/multiple GPUs` above.) Hence, the files will be overridden to save space when evaluating with a new `$EPOCH`.
- `dist_test_svm_pretrain.sh`: `SVM_WORK_DIR=$WORK_DIR/$PRETRAIN_NAME/`, e.g., if `PRETRAIN=pretrains/odc_r50_v1-5af5dd0c.pth`, then `PRETRAIN_NAME=odc_r50_v1-5af5dd0c.pth`; if `PRETRAIN=random`, then `PRETRAIN_NAME=random`.

Notes:
- The evaluation records are saved in `$SVM_WORK_DIR/logs/eval_svm.log`.
- When using `benchmarks/dist_test_svm_epoch.sh`, DO NOT launch multiple tests of the same experiment with different epochs, since they share the same working directory.
- Linear SVM takes 5 min, low-shot linear SVM takes about 1 hour with 32 CPU cores. If you want to save time, you may delete or comment the low-shot SVM testing command (the last line in the scripts).

### ImageNet / Places205 Linear Classification

**First**, extract backbone weights:
```shell
python tools/extract_backbone_weights.py ${CHECKPOINT} ${WEIGHT_FILE}
```
Arguments:
- `CHECKPOINTS`: the checkpoint file of a selfsup method named as `epoch_*.pth`.
- `WEIGHT_FILE`: the output backbone weights file, e.g., `pretrains/moco_r50_v1-4ad89b5c.pth`.

**Next**, train and test linear classification:
```shell
# train
bash benchmarks/dist_train_linear.sh ${CONFIG_FILE} ${WEIGHT_FILE} [optional arguments]
# test (unnecessary if have validation in training)
bash tools/dist_test.sh ${CONFIG_FILE} ${GPUS} ${CHECKPOINT}
```
Augments:
- `CONFIG_FILE`: Use config files under "configs/benchmarks/linear_classification/". Note that if you want to test DeepCluster that has a sobel layer before the backbone, you have to use the config file named `*_sobel.py`, e.g., `configs/benchmarks/linear_classification/imagenet/r50_multihead_sobel.py`.
- Optional arguments include:
    - `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
    - `--deterministic`: Switch on "deterministic" mode which slows down training but the results are reproducible.

Working directories:
Where are the checkpoints and logs? E.g., if you use `configs/benchmarks/linear_classification/imagenet/r50_multihead.py` to evaluate `pretrains/moco_r50_v1-4ad89b5c.pth`, then the working directories for this evalution is `work_dirs/benchmarks/linear_classification/imagenet/r50_multihead/moco_r50_v1-4ad89b5c.pth/`.

### ImageNet Semi-Supervised Classification

```shell
# train
bash benchmarks/dist_train_semi.sh ${CONFIG_FILE} ${WEIGHT_FILE} [optional arguments]
# test (unnecessary if have validation in training)
bash tools/dist_test.sh ${CONFIG_FILE} ${GPUS} ${CHECKPOINT}
```
Augments:
- `CONFIG_FILE`: Use config files under "configs/benchmarks/semi_classification/". Note that if you want to test DeepCluster that has a sobel layer before the backbone, you have to use the config file named `*_sobel.py`, e.g., `configs/benchmarks/semi_classification/imagenet_1percent/r50_sobel.py`.
- Optional arguments include:
    - `--resume_from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
    - `--deterministic`: Switch on "deterministic" mode which slows down training but the results are reproducible.

### VOC07+12 / COCO17 Object Detection

For more details to setup the environments for detection, please refer [here](https://github.com/open-mmlab/OpenSelfSup/blob/master/benchmarks/detection/README.md).

```shell
conda activate detectron2 # use detectron2 environment here, otherwise use open-mmlab environment
cd benchmarks/detection
python convert-pretrain-to-detectron2.py ${WEIGHT_FILE} ${OUTPUT_FILE} # must use .pkl as the output extension.
bash run.sh ${DET_CFG} ${OUTPUT_FILE}
```
Arguments:
- `WEIGHT_FILE`: The extracted backbone weights extracted aforementioned.
- `OUTPUT_FILE`: Converted backbone weights file, e.g., `odc_v1.pkl`.
- `DET_CFG`: The detectron2 config file, usually we use `configs/pascal_voc_R_50_C4_24k_moco.yaml`.

**Note**:
- This benchmark must use 8 GPUs as the default setting from MoCo.
- Please report the mean of 5 trials in your offical paper, according to MoCo.
- DeepCluster that uses Sobel layer is not supported by detectron2.

## Tools and Tips

### Count number of parameters

```shell
python tools/count_parameters.py ${CONFIG_FILE}
```

### Publish a model

Compute the hash of the weight file and append the hash id to the filename. The output file is the input file name with a hash suffix.

```shell
python tools/publish_model.py ${WEIGHT_FILE}
```
Arguments:
- `WEIGHT_FILE`: The extracted backbone weights extracted aforementioned.

### Reproducibility

If you want to make your performance exactly reproducible, please switch on `--deterministic` to train the final model to be published. Note that this flag will switch off `torch.backends.cudnn.benchmark` and slow down the training speed.

## How-to

### Use a new dataset

1. Write a data source file under `openselfsup/datasets/data_sources/`. You may refer to the existing ones.

2. Create new config files for your experiments.

### Design your own methods

#### What you need to do

    1. Create a dataset file under `openselfsup/datasets/` (better using existing ones);
    2. Create a model file under `openselfsup/models/`. The model typically contains:
      i) backbone (required): images to deep features from differet depth of layers. Your model must contain a `self.backbone` module, otherwise the backbone weights cannot be extracted.
      ii) neck (optional): deep features to compact feature vectors.
      iii) head (optional): define loss functions.
      iv) memory_bank (optional): define memory banks.
    3. Create a config file under `configs/` and setup the configs;
    4. [Optional] Create a hook file under `openselfsup/hooks/` if your method requires additional operations before run, every several iterations, every several epoch, or after run.
    
You may refer to existing modules under respective folders.

#### Features that may facilitate your implementation

* Decoupled data source and dataset.

Since dataset is correlated to a specific task while data source is general, we decouple data source and dataset in OpenSelfSup.

```python
data = dict(
    train=dict(type='ContrastiveDataset',
               data_source=dict(type='ImageNet', list_file='xx', root='xx'),
               pipeline=train_pipeline),
    val=dict(...),
    ...
)
```

* Configure data augmentations in the config file.

The augmentations are the same as `torchvision.transforms` except that `torchvision.transforms.RandomAppy` corresponds to `RandomAppliedTrans`. `Lighting` and `GaussianBlur` is additionally implemented.

```python
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

You may specify optimization paramters including lr, momentum and weight_decay for a certain group of paramters in the config file with `paramwise_options`. `paramwise_options` is a dict whose key is regular expressions and value is options. Options include 6 fields: lr, lr_mult, momentum, momentum_mult, weight_decay, weight_decay_mult, lars_exclude (only works with LARS optimizer).

```python
# this config sets all normalization layers with weight_decay_mult=0.1,
# and the head with `lr_mult=10, momentum=0`.
paramwise_options = {
    '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay_mult=0.1),
    '\Ahead.': dict(lr_mult=10, momentum=0)}
optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
                     weight_decay=0.0001,
                     paramwise_options=paramwise_options)
```

* Configure custom hooks in the config file.

The hooks will be called in order. For hook design, please refer to [odc_hook.py](https://github.com/open-mmlab/OpenSelfSup/blob/master/openselfsup/hooks/odc_hook.py) as an example.

```python
custom_hooks = [
    dict(type='DeepClusterHook', ...),
    dict(type='ODCHook', ...),
]
```
