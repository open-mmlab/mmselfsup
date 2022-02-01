# Changelog

## MMSelfSup

### v0.6.0 (02/02/2022)

#### Highlight
* Support vision transformer based MoCo v3 ([#194](https://github.com/open-mmlab/mmselfsup/pull/194))
* Speed up training and start time ([#181](https://github.com/open-mmlab/mmselfsup/pull/181))
* Support cpu training ([#188](https://github.com/open-mmlab/mmselfsup/pull/188))

#### New Features
* Support vision transformer based MoCo v3 ([#194](https://github.com/open-mmlab/mmselfsup/pull/194))
* Support cpu training ([#188](https://github.com/open-mmlab/mmselfsup/pull/188))

#### Bug Fixes
* Fix issue ([#159](https://github.com/open-mmlab/mmselfsup/issues/159), [#160](https://github.com/open-mmlab/mmselfsup/issues/160)) related bugs ([#161](https://github.com/open-mmlab/mmselfsup/pull/161))
* Fix missing prob assignment in `RandomAppliedTrans` ([#173](https://github.com/open-mmlab/mmselfsup/pull/173))
* Fix bug of showing k-means losses ([#182](https://github.com/open-mmlab/mmselfsup/pull/182))
* Fix bug in non-distributed multi-gpu training/testing ([#189](https://github.com/open-mmlab/mmselfsup/pull/189))
* Fix bug when loading cifar dataset ([#191](https://github.com/open-mmlab/mmselfsup/pull/191))
* Fix `dataset.evaluate` args bug ([#192](https://github.com/open-mmlab/mmselfsup/pull/192))

#### Improvements
* Cancel previous runs that are not completed in CI ([#145](https://github.com/open-mmlab/mmselfsup/pull/145))
* Enhance MIM function ([#152](https://github.com/open-mmlab/mmselfsup/pull/152))
* Skip CI when some specific files were changed ([#154](https://github.com/open-mmlab/mmselfsup/pull/154))
* Add `drop_last` when building eval optimizer ([#158](https://github.com/open-mmlab/mmselfsup/pull/158))
* Deprecate the support for "python setup.py test" ([#174](https://github.com/open-mmlab/mmselfsup/pull/174))
* Speed up training and start time ([#181](https://github.com/open-mmlab/mmselfsup/pull/181))
* Upgrade `isort` to 5.10.1 ([#184](https://github.com/open-mmlab/mmselfsup/pull/184))

#### Docs
* Refactor the directory structure of docs ([#146](https://github.com/open-mmlab/mmselfsup/pull/146))
* Fix readthedocs ([#148](https://github.com/open-mmlab/mmselfsup/pull/148), [#149](https://github.com/open-mmlab/mmselfsup/pull/149), [#153](https://github.com/open-mmlab/mmselfsup/pull/153))
* Fix typos and dead links in some docs ([#155](https://github.com/open-mmlab/mmselfsup/pull/155), [#180](https://github.com/open-mmlab/mmselfsup/pull/180), [#195](https://github.com/open-mmlab/mmselfsup/pull/195))
* Update training logs and benchmark results in model zoo ([#157](https://github.com/open-mmlab/mmselfsup/pull/157), [#165](https://github.com/open-mmlab/mmselfsup/pull/165), [#195](https://github.com/open-mmlab/mmselfsup/pull/195))
* Update and translate some docs into Chinese ([#163](https://github.com/open-mmlab/mmselfsup/pull/163), [#164](https://github.com/open-mmlab/mmselfsup/pull/164), [#165](https://github.com/open-mmlab/mmselfsup/pull/165), [#166](https://github.com/open-mmlab/mmselfsup/pull/166), [#167](https://github.com/open-mmlab/mmselfsup/pull/167), [#168](https://github.com/open-mmlab/mmselfsup/pull/168), [#169](https://github.com/open-mmlab/mmselfsup/pull/169), [#172](https://github.com/open-mmlab/mmselfsup/pull/172), [#176](https://github.com/open-mmlab/mmselfsup/pull/176), [#178](https://github.com/open-mmlab/mmselfsup/pull/178), [#179](https://github.com/open-mmlab/mmselfsup/pull/179))
* Update algorithm README with the new format ([#177](https://github.com/open-mmlab/mmselfsup/pull/177))


### v0.5.0 (16/12/2021)

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
* Add SwAV, SimSiam, DenseCL algorithms.
* Add t-SNE visualization tools.
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
