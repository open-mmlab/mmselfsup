# 常见问题解答

我们在这里列出了使用时的一些常见问题及其相应的解决方案。 如果您发现有一些问题被遗漏，请随时提 PR 丰富这个列表。 如果您无法在此获得帮助，请使用 [issue 模板](https://github.com/open-mmlab/mmselfsup/tree/master/.github/ISSUE_TEMPLATE)创建问题，但是请在模板中填写所有必填信息，这有助于我们更快定位问题。

## 安装相关

下表显示了与 MMSelfSup 适配的 MMCV, MMClassification, MMDetection 和 MMSegmentation 的版本号。 为避免安装过程中出现问题，请参照下表安装适配的版本。

| MMSelfSup version |    MMCV version     |  MMClassification version   | MMSegmentation version | MMDetection version |
| :---------------: | :-----------------: | :-------------------------: | :--------------------: | :-----------------: |
|  0.9.1 (master)   | mmcv-full >= 1.4.2  |       mmcls >= 0.21.0       |    mmseg >= 0.20.2     |   mmdet >= 2.19.0   |
|       0.9.0       | mmcv-full >= 1.4.2  |       mmcls >= 0.21.0       |    mmseg >= 0.20.2     |   mmdet >= 2.19.0   |
|       0.8.0       | mmcv-full >= 1.4.2  |       mmcls >= 0.21.0       |    mmseg >= 0.20.2     |   mmdet >= 2.19.0   |
|       0.7.1       | mmcv-full >= 1.3.16 | mmcls >= 0.19.0, \<= 0.20.1 |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |
|       0.6.0       | mmcv-full >= 1.3.16 |       mmcls >= 0.19.0       |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |
|       0.5.0       | mmcv-full >= 1.3.16 |              /              |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |

**注意:**

- 如果您已经安装了 mmcv, 您需要运行 `pip uninstall mmcv` 来卸载已经安装的 mmcv。 如果您在本地同时安装了 mmcv 和 mmcv-full, `ModuleNotFoundError` 将会抛出。
- 如过您仍然对版本问题有疑问，欢迎创建 issue 并提供您的依赖库信息。
