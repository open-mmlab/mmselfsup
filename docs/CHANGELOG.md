## Changelog

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
* Support updating network every several interations.
* Support LARS optimizer with nesterov.
* Support excluding specific parameters from LARS adaptation and weight decay required in SimCLR and BYOL.
