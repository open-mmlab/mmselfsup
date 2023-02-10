# FAQ

We list some common troubles faced by many users and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an issue using the [provided templates](https://github.com/open-mmlab/mmselfsup/tree/master/.github/ISSUE_TEMPLATE) and make sure you fill in all required information in the template.

- [FAQ](#faq)
  - [Installation](#installation)
  - [DeepCluster on A100 GPU](#deepcluster-on-a100-gpu)

## Installation

Compatible MMEngine, MMCV, MMClassification, MMDetection and MMSegmentation versions are shown below. Please install the correct version of them to avoid installation issues.

| MMSelfSup version |      MMEngine version       |        MMCV version        |  MMClassification version   | MMSegmentation version | MMDetection version |
| :---------------: | :-------------------------: | :------------------------: | :-------------------------: | :--------------------: | :-----------------: |
|  1.0.0rc6 (1.x)   | mmengine >= 0.4.0, \< 1.0.0 | mmcv >= 2.0.0rc1, \< 2.1.0 | mmcls >= 1.0.0rc5, \< 1.1.0 |   mmseg >= 1.0.0rc0    |  mmdet >= 3.0.0rc0  |
|     1.0.0rc5      | mmengine >= 0.4.0, \< 1.0.0 | mmcv >= 2.0.0rc1, \< 2.1.0 | mmcls >= 1.0.0rc5, \< 1.1.0 |   mmseg >= 1.0.0rc0    |  mmdet >= 3.0.0rc0  |
|     1.0.0rc4      | mmengine >= 0.3.0, \< 1.0.0 | mmcv >= 2.0.0rc1, \< 2.1.0 | mmcls >= 1.0.0rc4, \< 1.1.0 |   mmseg >= 1.0.0rc0    |  mmdet >= 3.0.0rc0  |
|     1.0.0rc3      | mmengine >= 0.3.0, \< 1.0.0 | mmcv >= 2.0.0rc1, \< 2.1.0 | mmcls >= 1.0.0rc0, \< 1.1.0 |   mmseg >= 1.0.0rc0    |  mmdet >= 3.0.0rc0  |
|     1.0.0rc2      |      mmengine >= 0.1.0      | mmcv >= 2.0.0rc1, \< 2.1.0 | mmcls >= 1.0.0rc0, \< 1.1.0 |   mmseg >= 1.0.0rc0    |  mmdet >= 3.0.0rc0  |
|     1.0.0rc1      |      mmengine >= 0.1.0      | mmcv >= 2.0.0rc1, \< 2.1.0 | mmcls >= 1.0.0rc0, \< 1.1.0 |   mmseg >= 1.0.0rc0    |  mmdet >= 3.0.0rc0  |
|       0.9.1       |              /              |     mmcv-full >= 1.4.2     |       mmcls >= 0.21.0       |    mmseg >= 0.20.2     |   mmdet >= 2.19.0   |
|       0.9.0       |              /              |     mmcv-full >= 1.4.2     |       mmcls >= 0.21.0       |    mmseg >= 0.20.2     |   mmdet >= 2.19.0   |
|       0.8.0       |              /              |     mmcv-full >= 1.4.2     |       mmcls >= 0.21.0       |    mmseg >= 0.20.2     |   mmdet >= 2.19.0   |
|       0.7.1       |              /              |    mmcv-full >= 1.3.16     | mmcls >= 0.19.0, \<= 0.20.1 |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |
|       0.6.0       |              /              |    mmcv-full >= 1.3.16     |       mmcls >= 0.19.0       |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |
|       0.5.0       |              /              |    mmcv-full >= 1.3.16     |              /              |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |

**Note:**

- MMDetection and MMSegmentation are optional.
- If you still have version problem, please create an issue and provide your package versions.

## DeepCluster on A100 GPU

Problem: If you want to try [DeepCluster](https://github.com/open-mmlab/mmselfsup/blob/master/configs/selfsup/deepcluster/README.md) algorithm on A100 GPU, use the `faiss` installed by pip will raise error, which is mentioned in [here](https://github.com/facebookresearch/faiss/issues/2076).

Please install `faiss` by conda like this:

```bash
conda install -c pytorch faiss-gpu cudatoolkit=11.3
```

> Also, you need to install PyTorch with the support of CUDA 11.3, and the faiss-gpu==1.7.2 requires python 3.6-3.8.
