# FAQ
我们列出来一些用户常见的问题，并将他们的解决方案列出。 您可以将一些您发现的常见的问题添加进列表中，来帮助其他用户解决问题。 如果这里面的内容没有覆盖您的问题，请按照 [provided templates](https://github.com/open-mmlab/mmselfsup/tree/master/.github/ISSUE_TEMPLATE) 创建一个 issue，并确保您在模板中填写了所有要求的信息。

- [FAQ](#faq)
  - [安装](#安装)
  - [A100 GPU DeepCluster](#A100 GPU DeepCluster)

## 安装
MMCV, MMClassification, MMDetection and MMSegmentation 的版本兼容性如下所示。 请安装正确的版本来避免安装问题。

| MMSelfSup version |    MMCV version     |  MMClassification version   | MMSegmentation version | MMDetection version |
| :---------------: | :-----------------: | :-------------------------: | :--------------------: | :-----------------: |
|  0.9.1 (master)   | mmcv-full >= 1.4.2  |       mmcls >= 0.21.0       |    mmseg >= 0.20.2     |   mmdet >= 2.19.0   |
|       0.9.0       | mmcv-full >= 1.4.2  |       mmcls >= 0.21.0       |    mmseg >= 0.20.2     |   mmdet >= 2.19.0   |
|       0.8.0       | mmcv-full >= 1.4.2  |       mmcls >= 0.21.0       |    mmseg >= 0.20.2     |   mmdet >= 2.19.0   |
|       0.7.1       | mmcv-full >= 1.3.16 | mmcls >= 0.19.0, \<= 0.20.1 |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |
|       0.6.0       | mmcv-full >= 1.3.16 |       mmcls >= 0.19.0       |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |
|       0.5.0       | mmcv-full >= 1.3.16 |              /              |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |

**Note:**

- MMDetection and MMSegmentation 是可选的。
- 如果您仍然有版本错误，请创建一个issue并提供您的包的版本信息。

## DeepCluster on A100 GPU
如果您想尝试 [DeepCluster](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/deepcluster/README.md) 在 A100 GPU上，使用 pip 安装 `faiss` 将会引发错误，
他在[这里](https://github.com/facebookresearch/faiss/issues/2076)被提及过。

请使用 conda 安装：

```bash
conda install -c pytorch faiss-gpu cudatoolkit=11.3
```
> 同时您需要安装支持 CUDA11.3 的 PyTorch，同时 faiss-gpu==1.7.2 要求 python 3.6-3.8。
