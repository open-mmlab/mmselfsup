# 结构

- [结构](#结构)
  - [SelfSupDataSample中的定制化的属性](#SelfSupDataSample中的定制化的属性)
  - [用MMSelfSup把数据打包给SelfSupDataSample](#用MMSelfSup把数据打包给SelfSupDataSample)

像OpenMMLab中其他仓库一样，MMSelfSup也定义了一个数据结构，名为`SelfSupDataSample`,这个数据结构用于接收和传递整个训练和测试过程中的数据。
`SelfSupDataSample`继承[MMEngine](https://github.com/open-mmlab/mmengine)中使用的`BaseDataElement`。如果需要深入了解`BaseDataElement`，我们建议参考[BaseDataElement](https://github.com/open-mmlab/mmengine/blob/main/docs/zh_cn/tutorials/data_element.md)。在这些教程中，我们主要讨论[SelfSupDataSample](mmselfsup.structures.SelfSupDataSample)中一些定制化的属性。

## SelfSupDataSample中的定制化的属性
在MMSelfSup中，`SelfSupDataSample`将模型需要的所有信息（除了图片）打包，比如mask image modeling(MIM)中请求的`mask`和前置任务中的`pseudo_label`。除了提供信息，它还能接受模型产生的信息，比如预测得分。为实现上述功能，`SelfSupDataSample`定义以下五个属性：

-gt_label（标签数据），包含图片的真实标签。

-sample_idx（例子的数据），包含一开始被数据集初始化的数据列表中的最近的图片的序号。

-mask（基础数据组成部分），包含MIM中的面具，比如SimMIM和CAE。

-pred_label（标签数据），包含模型预测的标签。

-pseudo_label（基础数据组成部分），包含前置任务中用到的假的标签，比如Relation Location中的location。

为了帮助使用者理解SelfSupDataSample中的基本思想，我们给出一个关于如何创建`SelfSupDataSample`实例并设置这些属性的简单例子。

```python
import torch
from mmselfsup.core import SelfSupDataSample
from mmengine.data import LabelData, InstanceData, BaseDataElement

selfsup_data_sample = SelfSupDataSample()
# set the gt_label in selfsup_data_sample
# gt_label should be the type of LabelData
selfsup_data_sample.gt_label = LabelData(value=torch.tensor([1]))

# setting gt_label to a type, which is not LabelData, will raise an error
selfsup_data_sample.gt_label = torch.tensor([1])
# AssertionError: tensor([1]) should be a <class 'mmengine.data.label_data.LabelData'> but got <class 'torch.Tensor'>

# set the sample_idx in selfsup_data_sample
# also, the assigned value of sample_idx should the type of InstanceData
selfsup_data_sample.sample_idx = InstanceData(value=torch.tensor([1]))

# setting the mask in selfsup_data_sample
selfsup_data_sample.mask = BaseDataElement(value=torch.ones((3, 3)))

# setting the pseudo_label in selfsup_data_sample
selfsup_data_sample.pseudo_label = InstanceData(location=torch.tensor([1, 2, 3]))


# After creating these attributes, you can easily fetch values in these attributes
print(selfsup_data_sample.gt_label.value)
# tensor([1])
print(selfsup_data_sample.mask.value.shape)
# torch.Size([3, 3])
```

## 用MMSelfSup把数据打包给SelfSupDataSample

在把数据喂给模型之前，MMSelfSup按照数据流程把数据打包进`SelfSupDataSample`。如果你不熟悉数据流程，可以参考
Before feeding data into model, MMSelfSup packs data into `SelfSupDataSample` in data pipeline.[data transform](https://github.com/open-mmlab/mmcv/blob/transforms/docs/zh_cn/understand_mmcv/data_transform.md). To pack data, we implement a data transform, called [PackSelfSupInputs](mmselfsup.datasets.transforms.PackSelfSupInputs)。我们用一个叫[PackSelfSupInputs](mmselfsup.datasets.transforms.PackSelfSupInputs)的数据转换来打包数据。

```python
class PackSelfSupInputs(BaseTransform):
    """Pack data into the format compatible with the inputs of algorithm.

    Required Keys:

    - img

    Added Keys:

    - data_sample
    - inputs

    Args:
        key (str): The key of image inputted into the model. Defaults to 'img'.
        algorithm_keys (List[str]): Keys of elements related
            to algorithms, e.g. mask. Defaults to [].
        pseudo_label_keys (List[str]): Keys set to be the attributes of
            pseudo_label. Defaults to [].
        meta_keys (List[str]): The keys of meta info of an image.
            Defaults to [].
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
        """Method to pack the data.

        Args:
            results (Dict): Result dict from the data pipeline.

        Returns:
            Dict:

            - 'inputs' (List[torch.Tensor]): The forward data of models.
            - 'data_sample' (SelfSupDataSample): The annotation info of the
                the forward data.
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

        # gt_label, sample_idx, mask, pred_label will be set here
        for key in self.algorithm_keys:
            self.set_algorithm_keys(data_sample, key, results)

        # keys, except for gt_label, sample_idx, mask, pred_label, will be
        # set as the attributes of pseudo_label
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
        """Set the algorithm keys of SelfSupDataSample."""
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

在SelfSupDataSample中`algorithm_keys`是除了`pseudo_label`的数据属性,`pseudo_label_keys`是SelfSupDataSample中假标签对应的分支属性。 
感谢读完整个教程。有问题的话可以在GitHub上提issue，我们会尽快联系你。
