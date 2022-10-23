# FAQ
我们列出来一些用户常见的问题，并将他们的解决方案列出。 您可以将一些您发现的常见的问题添加进列表中，来帮助其他用户解决问题。 如果这里面的内容没有覆盖您的问题，请按照 [provided templates](https://github.com/open-mmlab/mmselfsup/tree/master/.github/ISSUE_TEMPLATE) 创建一个 issue，并确保您在模板中填写了所有要求的信息。

- [FAQ](#faq)
  - [安装](#安装)

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

- 您事先需要运行 `pip uninstall mmcv` 如果您已经安装了 mmcv。 如果您同时安装了 mmcv 和 mmcv-full，将会有一个 `ModuleNotFoundError`。
- 如果您仍然有版本错误，请创建一个issue并提供您的包的版本信息。
