# 更新日志

## MMSelfSup

### v0.6.0 (02/02/2022)

#### 亮点
* 支持基于 vision transformer 的 MoCo v3 ([#194](https://github.com/open-mmlab/mmselfsup/pull/194))
* 加速训练和启动时间 ([#181](https://github.com/open-mmlab/mmselfsup/pull/181))
* 支持 cpu 训练 ([#188](https://github.com/open-mmlab/mmselfsup/pull/188))

#### 新特性
* 支持基于 vision transformer 的 MoCo v3 ([#194](https://github.com/open-mmlab/mmselfsup/pull/194))
* 支持 cpu 训练 ([#188](https://github.com/open-mmlab/mmselfsup/pull/188))

#### Bug 修复
* 修复问题 ([#159](https://github.com/open-mmlab/mmselfsup/issues/159), [#160](https://github.com/open-mmlab/mmselfsup/issues/160)) 中提到的相关 bugs ([#161](https://github.com/open-mmlab/mmselfsup/pull/161))
* 修复 `RandomAppliedTrans` 中缺失的 prob 赋值 ([#173](https://github.com/open-mmlab/mmselfsup/pull/173))
* 修复 k-means losses 显示的 bug ([#182](https://github.com/open-mmlab/mmselfsup/pull/182))
* 修复非分布式多 gpu 训练/测试中的 bug ([#189](https://github.com/open-mmlab/mmselfsup/pull/189))
* 修复加载 cifar 数据集时的 bug ([#191](https://github.com/open-mmlab/mmselfsup/pull/191))
* 修复 `dataset.evaluate` 的参数 bug ([#192](https://github.com/open-mmlab/mmselfsup/pull/192))

#### 改进
* 取消之前在 CI 中未完成的运行 ([#145](https://github.com/open-mmlab/mmselfsup/pull/145))
* 增强 MIM 功能 ([#152](https://github.com/open-mmlab/mmselfsup/pull/152))
* 更改某些特定文件时跳过 CI ([#154](https://github.com/open-mmlab/mmselfsup/pull/154))
* 在构建 eval 优化器时添加 `drop_last` 选项 ([#158](https://github.com/open-mmlab/mmselfsup/pull/158))
* 弃用对 “python setup.py test” 的支持 ([#174](https://github.com/open-mmlab/mmselfsup/pull/174))
* 加速训练和启动时间 ([#181](https://github.com/open-mmlab/mmselfsup/pull/181))
* 升级 `isort` 到 5.10.1 ([#184](https://github.com/open-mmlab/mmselfsup/pull/184))

#### 文档
* 重构文档目录结构 ([#146](https://github.com/open-mmlab/mmselfsup/pull/146))
* 修复 readthedocs ([#148](https://github.com/open-mmlab/mmselfsup/pull/148), [#149](https://github.com/open-mmlab/mmselfsup/pull/149), [#153](https://github.com/open-mmlab/mmselfsup/pull/153))
* 修复一些文档中的拼写错误和无效链接 ([#155](https://github.com/open-mmlab/mmselfsup/pull/155), [#180](https://github.com/open-mmlab/mmselfsup/pull/180), [#195](https://github.com/open-mmlab/mmselfsup/pull/195))
* 更新模型库里的训练日志和基准测试结果 ([#157](https://github.com/open-mmlab/mmselfsup/pull/157), [#165](https://github.com/open-mmlab/mmselfsup/pull/165), [#195](https://github.com/open-mmlab/mmselfsup/pull/195))
* 更新部分文档并翻译成中文 ([#163](https://github.com/open-mmlab/mmselfsup/pull/163), [#164](https://github.com/open-mmlab/mmselfsup/pull/164), [#165](https://github.com/open-mmlab/mmselfsup/pull/165), [#166](https://github.com/open-mmlab/mmselfsup/pull/166), [#167](https://github.com/open-mmlab/mmselfsup/pull/167), [#168](https://github.com/open-mmlab/mmselfsup/pull/168), [#169](https://github.com/open-mmlab/mmselfsup/pull/169), [#172](https://github.com/open-mmlab/mmselfsup/pull/172), [#176](https://github.com/open-mmlab/mmselfsup/pull/176), [#178](https://github.com/open-mmlab/mmselfsup/pull/178), [#179](https://github.com/open-mmlab/mmselfsup/pull/179))
* 更新算法 README 到新格式 ([#177](https://github.com/open-mmlab/mmselfsup/pull/177))


### v0.5.0 (16/12/2021)

#### 亮点
* 代码重构后发版。
* 添加 3 个新的自监督学习算法。
* 支持 MMDet 和 MMSeg 的基准测试。
* 添加全面的文档。

#### 重构
* 合并冗余数据集文件。
* 适配新版 MMCV，去除旧版相关代码。
* 继承 MMCV BaseModule。
* 优化目录结构。
* 重命名所有配置文件。

#### 新特性
* 添加 SwAV、SimSiam、DenseCL 算法。
* 添加 t-SNE 可视化工具。
* 支持 MMCV 版本 fp16。

#### 基准
* 更多基准测试结果，包括分类、检测和分割。
* 支持下游任务中的一些新数据集。
* 使用 MIM 启动 MMDet 和 MMSeg 训练。

#### 文档
* 重构 README、getting_started、install、model_zoo 文档。
* 添加数据准备文档。
* 添加全面的教程。


## OpenSelfSup (历史)

### v0.3.0 (14/10/2020)

#### 亮点
* 支持混合精度训练。
* 改进 GaussianBlur 使训练速度加倍。
* 更多基准测试结果。

#### Bug 修复
* 修复 moco v2 中的 bugs，现在结果可复现。
* 修复 byol 中的 bugs。

#### 新特性
* 混合精度训练。
* 改进 GaussianBlur 使 MoCo V2、SimCLR、BYOL 的训练速度加倍。
* 更多基准测试结果，包括 Places、VOC、COCO。

### v0.2.0 (26/6/2020)

#### 亮点
* 支持 BYOL。
* 支持半监督基准测试。

#### Bug 修复
* 修复 publish_model.py 中的哈希 id。

#### 新特性

* 支持 BYOL。
* 在线性和半监督评估中将训练和测试脚本分开。
* 支持半监督基准测试：benchmarks/dist_train_semi.sh。
* 将基准测试相关的配置文件移动到 configs/benchmarks/。
* 提供基准测试结果和模型下载链接。
* 支持每隔几次迭代更新网络。
* 支持带有 Nesterov 的 LARS 优化器。
* 支持 SimCLR 和 BYOL 从 LARS 适应和权重衰减中排除特定参数的需求。
