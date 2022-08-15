# FAQ

We list some common troubles faced by many users and their corresponding solutions here. Feel free to enrich the list if you find any frequent issues and have ways to help others to solve them. If the contents here do not cover your issue, please create an issue using the [provided templates](https://github.com/open-mmlab/mmselfsup/tree/master/.github/ISSUE_TEMPLATE) and make sure you fill in all required information in the template.

## Installation

Compatible MMCV, MMClassification, MMDetection and MMSegmentation versions are shown below. Please install the correct version of them to avoid installation issues.

| MMSelfSup version |    MMCV version     |  MMClassification version   | MMSegmentation version | MMDetection version |
| :---------------: | :-----------------: | :-------------------------: | :--------------------: | :-----------------: |
|  0.9.1 (master)   | mmcv-full >= 1.4.2  |       mmcls >= 0.21.0       |    mmseg >= 0.20.2     |   mmdet >= 2.19.0   |
|       0.9.0       | mmcv-full >= 1.4.2  |       mmcls >= 0.21.0       |    mmseg >= 0.20.2     |   mmdet >= 2.19.0   |
|       0.8.0       | mmcv-full >= 1.4.2  |       mmcls >= 0.21.0       |    mmseg >= 0.20.2     |   mmdet >= 2.19.0   |
|       0.7.1       | mmcv-full >= 1.3.16 | mmcls >= 0.19.0, \<= 0.20.1 |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |
|       0.6.0       | mmcv-full >= 1.3.16 |       mmcls >= 0.19.0       |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |
|       0.5.0       | mmcv-full >= 1.3.16 |              /              |    mmseg >= 0.20.2     |   mmdet >= 2.16.0   |

**Note:**

- You need to run `pip uninstall mmcv` first if you have mmcv installed. If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.
- If you still have version problem, please create an issue and provide your package versions.
