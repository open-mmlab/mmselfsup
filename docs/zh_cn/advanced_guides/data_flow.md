# 数据流

- [数据流](#数据流)
  - [数据加载器与模型之间的数据流](#数据加载器与模型之间的数据流)
    - [数据集的数据处理](#数据集的数据处理)
    - [数据加载器的数据处理](#数据加载器的数据处理)
    - [数据预处理器的数据处理](#数据预处理器的数据处理)

数据流（Data Flow）定义了数据在两个独立模块之间传递的方式，如数据加载器（dataloader）模块与模型（model）模块，如下图所示。

<div align="left">
<img src="https://user-images.githubusercontent.com/30762564/185855134-89f5be9e-39ca-4da4-bd87-7cf26e80ab2f.png" width="70%"/>
</div>

在 MMSelfSup 中，我们主要关注两类数据流，一是数据加载器 （dataloader）与模型（model）之间，二是模型与可视化工具（visualizer）之间。 而对于模型与 metric 之间数据流的介绍，大家可以参考 OpenMMLab 其他代码库中的文档，如  [MMClassification](https://github.com/open-mmlab/mmclassification). 此外，对于 model 与 visualizer 模块之间的数据流，感兴趣的话可以参考： [visualization](../user_guides/visualization.md).

## 数据加载器与模型之间的数据流

数据加载器 （dataloader） 和模型 （model）之间的数据流一般可以分为如下三个步骤 :

i) 使用 `PackSelfSupInputs` 将转换完成的数据打包成为一个字典;

ii) 使用 `collate_fn` 将各个张量集成为一个批处理张量;

iii) 数据预处理器把以上所有数据迁移到 GPUS 等目标设备，并在数据加载器中将之前打包的字典解压为一个元组，该元祖包含输入图像与对应的元信息（`SelfSupDataSample`）。

### 数据集的数据处理

在 MMSelfSup 中，数据在投入到模型中前，会先进行一系列转换，称为`pipeline`，如常用的 `RandomResizedCrop` 和 `ColorJitter`转换。 在`pipeline`中完成若干次转换后，最后一步转换是`PackSelfSupInputs`， `PackSelfSupInputs` 会将转换好的数据打包到一个字典中，此字典包含两部分，即 `inputs` 和 `data_samples`.

```python
# 在这部分，我们省略了一些不太重要的代码

class PackSelfSupInputs(BaseTransform):

    def transform(self,
                  results: Dict) -> Dict[torch.Tensor, SelfSupDataSample]:

        packed_results = dict()
        if self.key in results:
            ...
            packed_results['inputs'] = img

        ...
        packed_results['data_samples'] = data_sample

        return packed_results
```

提示：`inputs` 包含了一个图像列表，例如一个应用在对比学习中的多视图列表。 即使输入是单个视图，`PackSelfSupInputs` 仍然会把信息输出到一个列表中。

### 数据加载器的数据处理

以数据集中的获取字典列表作为输入，数据加载器（dataloader）中的 `collect_fn` 会提取每个字典的`inputs`并将其整合成一个批处理张量；此外，每个字典中的`data_sample`也会被整合为一个列表，从而输出一个与先前字典有相同键的字典；最终数据加载器会通过 `collect_fn` 输出这个字典。

### 数据预处理器的数据处理

数据预处理是数据输入模型之前，处理数据过程的最后一步。 数据预处理过程会对图像进行归一处理，如把 BGR 模式转换为 RGB 模式，并将所有数据迁移至 GPU 等目标设备中 。上述各步骤完成后，最终会得到一个元组，该元组包含一个批处理图像的列表，和一个数据样本的列表。

```python
class SelfSupDataPreprocessor(ImgDataPreprocessor):

    def forward(
            self,
            data: dict,
            training: bool = False
    ) -> Tuple[List[torch.Tensor], Optional[list]]:

        assert isinstance(data,
                          dict), 'Please use default_collate in dataloader, \
            instead of pseudo_collate.'

        data = [val for _, val in data.items()]
        batch_inputs, batch_data_samples = self.cast_data(data)
        # channel transform
        if self._channel_conversion:
            batch_inputs = [
                _input[:, [2, 1, 0], ...] for _input in batch_inputs
            ]

        # 转换为 float 格式
        # 以保障效率
        batch_inputs = [input_.float() for input_ in batch_inputs]

        # 该步骤为归一化。 这与 :class:`mmengine.ImgDataPreprocessor` 有所不同。
        # 由于某些算法（如 SimCLR ）的图像有多个视图，所以输入中的每项都是一个列表，
        # 其中包含一张图像的多个视图。
        if self._enable_normalize:
            batch_inputs = [(_input - self.mean) / self.std
                            for _input in batch_inputs]

        return batch_inputs, batch_data_samples
```
