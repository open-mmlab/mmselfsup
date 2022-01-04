# 变更日志

## MMSelfSup

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
