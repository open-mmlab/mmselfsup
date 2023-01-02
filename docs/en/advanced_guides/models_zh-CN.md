# 模型

- [模型](#models)
  - [MMSelfSup模型概述](#MMSelfSup模型概述)
  - [用子模块来构造算法](#用子模块来构造算法)
  - [纵览基础模型中的抽象函数](#纵览基础模型中的抽象函数)

我们可以把模型看作算法的特征提取器或者损失生成器。在MMSelfSup中，模型主要包括以下几个部分:

-算法，包括模型的全部模块和构造算法时需要用到的子模块。
-脊柱，里面是每个算法的支柱，比如MAE中的VIT和SimMIM中的Swin Transformer。
-颈部，指一些特殊的模块，比如解码器，它直接增加脊柱部分的输出结果。
-头部，指一些特殊的模块，比如多层感知器的层，它增加脊柱部分或者颈部部分的输出结果。
-记忆，也就是一些算法中的存储体或者队列，比如MoCov1/v2。
-损失，用于算预测的输出和目标之间的损失。

## MMSelfSup模型概述

首先，我们概览MMSelfSup中已有的模型。我们根据上述的分类来展示这些模型。

|       算法       |            脊柱             |             颈部             |                 头部                 |                损失                |         记忆         |
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

正如上表所述，每个算法都是由脊柱，颈部，头部，损失和记忆的结合体。你可以任意从这些模块中选出一部分来构建你自己的算法。如果需要定制化的模块，你可借助[add_modules](./add_modules.md)来满足你的需求。
MMSelfSup提供一个基础模型,这个模型叫`BaseModel`, 所以的算法都应该继承这个基础模型。而且所有子模块（除了记忆部分）在算法初始化时都基于这个基础模型。记忆部分在每个算法的'__init__'中被构造。损失部分在构造头部部分时被构造。

```python
class BaseModel(_BaseModel):

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
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

正如上面代码所示，构造脊柱部分时需要配置，但是对颈部和头部而言这是可有可无。除了构造算法之外，你还需要重写基础模型中的一些抽象函数才能得到正确结果，我们将在下一部分讨论这件事。

## 纵览基础模型中的抽象函数
'forward'函数是结果的入口。然而，它和大多数Pytorch代码中只有一种模式的'forward'函数不同。在'forward'函数中你会逻辑混乱，这会限制粗糙度。正如下面代码所示，MMSelfSup中的'forward'函数有三种模式：张量，损失和预测。

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

-张量，如果模式为`tensor`，'forward'函数就返回从图片提取到的特征。你应该重写其中的'extract_feat'部分才能让定制化的提取过程有效。

-损失，如果模式为`loss`，'forward'函数就返回预测值与目标之间的损失。同样的，你应该重写其中的'extract_feat'部分才能让定制化的提取过程有效。

-预测，如果模式为`predict`，'forward'函数就返回预测结果，比如用你的算法预测的标签。如果需要，`predict`函数也需要重写。

现在我们学习了MMSelfSup中的模型的基本组成部分，如果你想深入研究，可以参考每个算法的API文件。
