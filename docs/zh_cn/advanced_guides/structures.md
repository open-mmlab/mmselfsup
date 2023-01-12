# 数据结构

- [数据结构](#数据结构)
  - [SelfSupDataSample 中的定制化的属性](#selfsupdatasample-中的定制化的属性)
  - [用 MMSelfSup 把数据打包给 SelfSupDataSample](#用-mmselfsup-把数据打包给-selfsupdatasample)

像 OpenMMLab 中其他仓库一样，MMSelfSup 也定义了一个数据结构，名为 `SelfSupDataSample` ,这个数据结构用于接收和传递整个训练和测试过程中的数据。
`SelfSupDataSample` 继承 [MMEngine](https://github.com/open-mmlab/mmengine) 中使用的 `BaseDataElement`。如果需要深入了解 `BaseDataElement`，我们建议参考 [BaseDataElement](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/advanced_tutorials/data_element.md)。在这些教程中，我们主要讨论 [SelfSupDataSample](mmselfsup.structures.SelfSupDataSample) 中一些定制化的属性。

## SelfSupDataSample 中的定制化的属性

在 MMSelfSup 中，`SelfSupDataSample` 将模型需要的所有信息（除了图片）打包，比如 mask image modeling(MIM) 中请求的 `mask` 和前置任务中的 `pseudo_label` 。除了提供信息，它还能接受模型产生的信息，比如预测得分。为实现上述功能， `SelfSupDataSample` 定义以下五个属性：

- gt_label（标签数据），包含图片的真实标签。

- sample_idx（实例数据），包含一开始被数据集初始化的数据列表中的最近的图片的序号。

- mask（数据基类），包含 MIM 中的面具，比如 SimMIM 和 CAE。

- pred_label（标签数据），包含模型预测的标签。

- pseudo_label（数据基类），包含前置任务中用到的假的标签，比如 Relation Location 中的 location。

为了帮助使用者理解 SelfSupDataSample 中的基本思想，我们给出一个关于如何创建 `SelfSupDataSample` 实例并设置这些属性的简单例子。

```python
import torch
from mmselfsup.core import SelfSupDataSample
from mmengine.data import LabelData, InstanceData, BaseDataElement

selfsup_data_sample = SelfSupDataSample()
# 在 selfsup_data_sample 里加入真实标签数据
# 真实标签数据的类型应与 LabelData 的类型一致
selfsup_data_sample.gt_label = LabelData(value=torch.tensor([1]))

# 如果真实标签数据类型和 LabelData 不一致会报错
selfsup_data_sample.gt_label = torch.tensor([1])
# 报错： AssertionError: tensor([1]) should be a <class 'mmengine.data.label_data.LabelData'> but got <class 'torch.Tensor'>

# 给 selfsup_data_sample 加入样例数据
# 同样的，样例数据里的值的类型应与 InstanceData 保持一致
selfsup_data_sample.sample_idx = InstanceData(value=torch.tensor([1]))

# 给 selfsup_data_sample 加面具
selfsup_data_sample.mask = BaseDataElement(value=torch.ones((3, 3)))

# 给 selfsup_data_sample 加假标签
selfsup_data_sample.pseudo_label = InstanceData(location=torch.tensor([1, 2, 3]))


# 创建这些属性后，您可轻而易举得取这些属性里的值
print(selfsup_data_sample.gt_label.value)
# 输出 tensor([1])
print(selfsup_data_sample.mask.value.shape)
# 输出 torch.Size([3, 3])
```

## 用 MMSelfSup 把数据打包给 SelfSupDataSample

在把数据喂给模型之前， MMSelfSup 按照数据流程把数据打包进 `SelfSupDataSample` 。如果您不熟悉数据流程，可以参考 [data transform](https://github.com/open-mmlab/mmcv/blob/transforms/docs/zh_cn/understand_mmcv/data_transform.md)。我们用一个叫 [PackSelfSupInputs](mmselfsup.datasets.transforms.PackSelfSupInputs)的数据变换来打包数据。

```python
class PackSelfSupInputs(BaseTransform):
    """把数据打包并让格式能与函数输入匹配

    需要的值：

    - img

    添加的值：

    - data_sample
    - inputs

    参数:
        key (str): 输入模型的图片的值，默认为 img 。
        algorithm_keys (List[str]): 和算法相关的组成部分的值，比如 mask 。默认为 [] 。
        pseudo_label_keys (List[str]): 假标签对应的属性。默认为 [] 。
        meta_keys (List[str]): 图片的 meta 信息的值。默认为 [] 。

    """

    def __init__(self,
                 key: Optional[str] = 'img',
                 algorithm_keys: Optional[List[str]] = [],
                 pseudo_label_keys: Optional[List[str]] = [],
                 meta_keys: Optional[List[str]] = []) -> None:
        assert isinstance(key, str), f'key should be the type of str, instead \
            of {type(key)}.'

        self.key = key
        self.algorithm_keys = algorithm_keys
        self.pseudo_label_keys = pseudo_label_keys
        self.meta_keys = meta_keys

    def transform(self,
                  results: Dict) -> Dict[torch.Tensor, SelfSupDataSample]:
        """打包数据的方法。

        参数:
            results (Dict): 数据变换返回的字典。

        返回:
            Dict:

            - 'inputs' (List[torch.Tensor]): 模型前面的数据。
            - 'data_sample' (SelfSupDataSample): 前面数据的注释信息。
        """
        packed_results = dict()
        if self.key in results:
            img = results[self.key]
            # if img is not a list, convert it to a list
            if not isinstance(img, List):
                img = [img]
            for i, img_ in enumerate(img):
                if len(img_.shape) < 3:
                    img_ = np.expand_dims(img_, -1)
                img_ = np.ascontiguousarray(img_.transpose(2, 0, 1))
                img[i] = to_tensor(img_)
            packed_results['inputs'] = img

        data_sample = SelfSupDataSample()
        if len(self.pseudo_label_keys) > 0:
            pseudo_label = InstanceData()
            data_sample.pseudo_label = pseudo_label

        # gt_label, sample_idx, mask, pred_label 在此设置
        for key in self.algorithm_keys:
            self.set_algorithm_keys(data_sample, key, results)

        # 除 gt_label, sample_idx, mask, pred_label 外的值会被设为假标签的属性
        for key in self.pseudo_label_keys:
            # convert data to torch.Tensor
            value = to_tensor(results[key])
            setattr(data_sample.pseudo_label, key, value)

        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_sample'] = data_sample

        return packed_results

    @classmethod
    def set_algorithm_keys(self, data_sample: SelfSupDataSample, key: str,
                           results: Dict) -> None:
        """设置 SelfSupDataSample 中算法的值."""
        value = to_tensor(results[key])
        if key == 'sample_idx':
            sample_idx = InstanceData(value=value)
            setattr(data_sample, 'sample_idx', sample_idx)
        elif key == 'mask':
            mask = InstanceData(value=value)
            setattr(data_sample, 'mask', mask)
        elif key == 'gt_label':
            gt_label = LabelData(value=value)
            setattr(data_sample, 'gt_label', gt_label)
        elif key == 'pred_label':
            pred_label = LabelData(value=value)
            setattr(data_sample, 'pred_label', pred_label)
        else:
            raise AttributeError(f'{key} is not a attribute of \
                SelfSupDataSample')
```

在 SelfSupDataSample 中 `algorithm_keys` 是除了 `pseudo_label` 的数据属性, `pseudo_label_keys` 是 SelfSupDataSample 中假标签对应的分支属性。
感谢读完整个教程。有问题的话可以在 GitHub 上提 issue，我们会尽快联系您。
