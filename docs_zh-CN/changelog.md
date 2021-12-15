# Changelog

## MMSelfSup

### v0.4.0 (13/12/2021)

#### Highlight
* Released with code refactor.
* Add 3 new self-supervised learning algorithms.
* Support benchmarks with MMDet and MMSeg.
* Add comprehensive documents.

#### Refactor
* Merge redundant dataset files.
* Adapt to new version of MMCV and remove old version related codes.
* Inherit MMCV BaseModule.
* Optimize directory.
* Rename all config files.

#### New Features
* Add SwAV, SimSiam, DenseCL algorithm.
* Add tsne visualization tools.
* Support MMCV version fp16.

#### Benchmarks
* More benchmarking results, including classification, detection and segmentation.
* Support some new datasets in downstream tasks.
* Launch MMDet and MMSeg training with MIM.

#### Docs
* Refactor README, getting_started, install, model_zoo files.
* Add data_prepare file.
* Add comprehensive tutorials.


## OpenSelfSup (History)

### v0.3.0 (14/10/2020)

#### Highlight
* Support Mixed Precision Training
* Improvement of GaussianBlur doubles the training speed
* More benchmarking results

#### Bug Fixes
* Fix bugs in moco v2, now the results are reproducible.
* Fix bugs in byol.

#### New Features
* Mixed Precision Training
* Improvement of GaussianBlur doubles the training speed of MoCo V2, SimCLR, BYOL
* More benchmarking results, including Places, VOC, COCO

### v0.2.0 (26/6/2020)

#### Highlights
* Support BYOL
* Support semi-supervised benchmarks

#### Bug Fixes
* Fix hash id in publish_model.py

#### New Features

* Support BYOL.
* Separate train and test scripts in linear/semi evaluation.
* Support semi-supevised benchmarks: benchmarks/dist_train_semi.sh.
* Move benchmarks related configs into configs/benchmarks/.
* Provide benchmarking results and model download links.
* Support updating network every several iterations.
* Support LARS optimizer with nesterov.
* Support excluding specific parameters from LARS adaptation and weight decay required in SimCLR and BYOL.
