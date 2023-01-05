# 模型

- [模型](#模型)
  - [MMSelfSup 模型概述](#mmselfsup-模型概述)
  - [用子模块来构造算法](#用子模块来构造算法)
  - [基础模型中的抽象函数](#基础模型中的抽象函数)

我们可以把模型看作算法的特征提取器或者损失生成器。在 MMSelfSup 中，模型主要包括以下几个部分:

- 算法，包括模型的全部模块和构造算法时需要用到的子模块。

- 主干，里面是每个算法的支柱，比如 MAE 中的 VIT 和 SimMIM 中的 Swin Transformer。

- 颈部，指一些特殊的模块，比如解码器，它直接增加脊柱部分的输出结果。

- 头部，指一些特殊的模块，比如多层感知器的层，它增加脊柱部分或者颈部部分的输出结果。

- 记忆，也就是一些算法中的存储体或者队列，比如 MoCo v1/v2。

- 损失，用于算输出的预测值和目标之间的损失。

- 目标生成器，为自监督学习生成优化目标，例如 HOG，其它模块抽取的特征（DALL-E，CLIP）等.

## MMSelfSup 模型概述

首先，我们纵览 MMSelfSup 中已有的模型。我们根据上述的分类来展示这些模型。

|          算法          |              主干               |             颈部             |                 头部                 |                损失                |          记忆          |
| :--------------------: | :-----------------------------: | :--------------------------: | :----------------------------------: | :--------------------------------: | :--------------------: |
| [`BarlowTwins`](TODO)  |        [`ResNet`](TODO)         |   [`NonLinearNeck`](TODO)    | [`LatentCrossCorrelationHead`](TODO) |   [`CrossCorrelationLoss`](TODO)   |          N/A           |
|   [`DenseCL`](TODO)    |        [`ResNet`](TODO)         |    [`DenseCLNeck`](TODO)     |      [`ContrastiveHead`](TODO)       |     [`CrossEntropyLoss`](TODO)     |          N/A           |
|     [`BYOL`](TODO)     |        [`ResNet`](TODO)         |   [`NonLinearNeck`](TODO)    |     [`LatentPredictHead`](TODO)      |   [`CosineSimilarityLoss`](TODO)   |          N/A           |
|     [`CAE`](TODO)      |        [`CAEViT`](TODO)         |      [`CAENeck`](TODO)       |          [`CAEHead`](TODO)           |         [`CAELoss`](TODO)          |          N/A           |
| [`DeepCluster`](TODO)  |        [`ResNet`](TODO)         |   [`AvgPool2dNeck`](TODO)    |          [`ClsHead`](TODO)           |     [`CrossEntropyLoss`](TODO)     |          N/A           |
|     [`MAE`](TODO)      |        [`MAEViT`](TODO)         | [`MAEPretrainDecoder`](TODO) |      [`MAEPretrainHead`](TODO)       |  [`MAEReconstructionLoss`](TODO)   |          N/A           |
|     [`MoCo`](TODO)     |        [`ResNet`](TODO)         |     [`LinearNeck`](TODO)     |      [`ContrastiveHead`](TODO)       |     [`CrossEntropyLoss`](TODO)     |          N/A           |
|    [`MoCov3`](TODO)    |       [`MoCoV3ViT`](TODO)       |   [`NonLinearNeck`](TODO)    |         [`MoCoV3Head`](TODO)         |     [`CrossEntropyLoss`](TODO)     |          N/A           |
|     [`NPID`](TODO)     |        [`ResNet`](TODO)         |     [`LinearNeck`](TODO)     |      [`ContrastiveHead`](TODO)       |     [`CrossEntropyLoss`](TODO)     | [`SimpleMemory`](TODO) |
|     [`ODC`](TODO)      |        [`ResNet`](TODO)         |      [`ODCNeck`](TODO)       |          [`ClsHead`](TODO)           |     [`CrossEntropyLoss`](TODO)     |  [`ODCMemory`](TODO)   |
| [`RelativeLoc`](TODO)  |        [`ResNet`](TODO)         |  [`RelativeLocNeck`](TODO)   |          [`ClsHead`](TODO)           |     [`CrossEntropyLoss`](TODO)     |          N/A           |
| [`RotationPred`](TODO) |        [`ResNet`](TODO)         |             N/A              |          [`ClsHead`](TODO)           |     [`CrossEntropyLoss`](TODO)     |          N/A           |
|    [`SimCLR`](TODO)    |        [`ResNet`](TODO)         |   [`NonLinearNeck`](TODO)    |      [`ContrastiveHead`](TODO)       |     [`CrossEntropyLoss`](TODO)     |          N/A           |
|    [`SimMIM`](TODO)    | [`SimMIMSwinTransformer`](TODO) |     [`SimMIMNeck`](TODO)     |         [`SimMIMHead`](TODO)         | [`SimMIMReconstructionLoss`](TODO) |          N/A           |
|   [`SimSiam`](TODO)    |        [`ResNet`](TODO)         |   [`NonLinearNeck`](TODO)    |     [`LatentPredictHead`](TODO)      |   [`CosineSimilarityLoss`](TODO)   |          N/A           |
|     [`SwAV`](TODO)     |        [`ResNet`](TODO)         |      [`SwAVNeck`](TODO)      |          [`SwAVHead`](TODO)          |         [`SwAVLoss`](TODO)         |          N/A           |

## 用子模块来构造算法

正如上表所述，每个算法都是主干，颈部，头部，损失和记忆的结合体。您可以从这些模块中任意选出若干部分来构建你自己的算法。如果需要定制化的模块，您可参考 [add_modules](./add_modules.md) 中的内容。
MMSelfSup 提供一个基础模型，名为 `BaseModel`，所以的算法都应该继承这个基础模型，而且所有子模块（除了记忆部分）在基础模型中进行初始化。记忆部分在对应算法的 `__init__` 中被构造。损失部分在头部部分初始化时被构造。

```python
class BaseModel(_BaseModel):

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 target_generator: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):

        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type',
                                     'mmselfsup.SelfSupDataPreprocessor')

        super().__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if head is not None:
            self.head = MODELS.build(head)

```

正如上面代码所示，构造主干部分时需要配置，但是对颈部和头部而言这可有可无。除了构造算法之外，您还需要重写基础模型中的一些抽象函数才能得到正确结果，我们将在下一部分讨论这件事。

## 基础模型中的抽象函数

`forward` 函数是结果的入口。然而，它和大多数 Pytorch 代码中只有一种模式的 `forward` 函数不同。MMSelfSup 把所有的逻辑都混杂在 `forward` 中，从而限制了该方法的可拓展性。正如下面代码所示，MMSelfSup 中的 `forward` 函数根据不同模式进行前向处理，目前共有三种模式：张量，损失和预测。

```python
def forward(self,
                batch_inputs: torch.Tensor,
                data_samples: Optional[List[SelfSupDataSample]] = None,
                mode: str = 'tensor'):
    if mode == 'tensor':
        feats = self.extract_feat(batch_inputs)
        return feats
    elif mode == 'loss':
        return self.loss(batch_inputs, data_samples)
    elif mode == 'predict':
        return self.predict(batch_inputs, data_samples)
    else:
        raise RuntimeError(f'Invalid mode "{mode}".')
```

- 张量，如果模式为 `tensor`，`forward` 函数就返回从图片提取到的特征。您应该重写其中的  `extract_feat`部分才能让定制化的提取过程有效。

- 损失，如果模式为 `loss`，`forward` 函数就返回预测值与目标之间的损失。同样的，您应该重写其中的 `loss` 部分才能让定制化的提取过程有效。

- 预测，如果模式为 `predict`，`forward` 函数就返回预测结果，比如用您的算法预测得到的标签。如果需要，`predict`函数也需要重写。

本文中我们学习了 MMSelfSup 中的模型的基本组成部分，如果您想深入研究，可以参考每个算法的API文件。
