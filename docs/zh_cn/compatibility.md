# MMSelfSup 和 OpenSelfSup 的不同点

该文件记录了最新版本 MMSelfSup 和 旧版以及 OpenSelfSup之间的区别。

MMSelfSup 进行了重构并解决了许多遗留问题，它与 OpenSelfSup 并不兼容，如旧的配置文件需要更新，因为某些类或组件的名称已被修改。

主要的不同点为：代码库约定，模块化设计。

## 模块化设计

为了构建更加清晰的目录结构， MMSelfSup 重新设计了一些模块。

### 数据集

- MMSelfSup 合并了部分数据集，减少了冗余代码.

  - Classification, Extraction, NPID -> OneViewDataset

  - BYOL, Contrastive -> MultiViewDataset

- 重构了 `data_sources` 文件夹，现在的数据读取函数更加鲁棒。

另外，该部分仍然在重构中，会在接下里的某一版本中发布。

### 模型

- 注册机制已经更新。 现在，`models` 文件夹下的各部分在构建时会有一个从 `MMCV` 中引入的父类 `MMCV_MODELS`。请查阅 [mmselfsup/models/builder.py](https://github.com/open-mmlab/mmselfsup/blob/master/mmselfsup/models/builder.py) 和 [mmcv/utils/registry.py](https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py) 获取更多信息。

- `models` 文件夹包含 `algorithms`，`backbones`，`necks`，`heads`，`memories` 和一些所需的工具。`algorithms` 部分集成了其他主要的组件来构建自监督学习算法，就像 `MMCls` 中的 `classifiers` 或者 `MMDet` 中的 `detectors`。

- 在 OpenSelfSup 中，`necks` 的命名会有一些混乱并且实现在同一个 python 文件中。现在，`necks` 部分已经被重构，通过一个文件进行归类管理，并进行重新命名。请查阅 `mmselfsup/models/necks` 获取更多信息。

## 代码库约定

由于 OpenSelfSup 很久没有更新，MMSelfSup 更新了代码库约定。

### 配置文件

- MMSelfSup 对所有配置文件进行了重命名，并制定了命名规范，请查阅 [0_config](./tutorials/0_config.md) 获取更多信息。

- 在配置文件中，一些类的参数命名或组件名字已被修改。

  - 一个算法名被修改： MOCO -> MoCo

  - 由于所有模型的组件继承自 `MMCV` 的 `BaseModule`，模型根据 `init_cfg` 进行初始化。请您依照该规范进行初始化设置，`init_weights` 仍然适用。

  - 请使用新的 necks 的命名来组合算，请在写配置文件前确认。

  - 归一化层通过 `norm_cfg` 进行管理。

### 脚本

- `tools` 的目录结构被修改，现在更加清晰，通过多个文件夹进行脚本分类和管理。 例如，两个转换文件夹分为为了模型和数据格式。 另外，和基准测试相关的脚本都在 `benchmarks` 文件夹中，它和 `configs/benchmarks` 拥有相同的目录结构。

- `train.py` 脚本中的参数已被更新， 两个主要修改点为

  - 增加 `--cfg-options` 参数，通过命令行传参对配置文件进行修改。

  - 移除 `--pretrained`， 通过 `--cfg-options` 设置预训练模型。
