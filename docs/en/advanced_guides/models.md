# Models

- [Models](#models)
  - [Overview of modules in MMSelfSup](#overview-of-modules-in-mmselfsup)
  - [Construct algorithms from sub-modules](#construct-algorithms-from-sub-modules)
  - [Overview these abstract functions in base model](#overview-these-abstract-functions-in-base-model)

Model can be seen as a feature extractor or loss generator for each algorithm. In MMSelfSup, it mainly
contains the following fix parts,

- algorithms, containing the full modules of a model and all sub-modules will be
  constructed in algorithms.

- backbones, containing the backbones for each algorithm, e.g. ViT for MAE, and Swim Transformer for SimMIM.

- necks, some specifial modules, such as decoder, appended directly to the output of the backbone.

- heads, some specifial modules, such as mlp layers, appended to the output of the backbone or neck.

- memories, some memory banks or queues in some algorithms, e.g. MoCo v1/v2.

- losses, used to compute the loss between the predicted output and the target.

- target_generators, generating targets for self-supervised learning optimization, such as HOG, extracted features from other modules(DALL-E, CLIP), etc.

## Overview of modules in MMSelfSup

First, we will give an overview about existing modules in MMSelfSup. They will be displayed according to the categories
described above.

|       algorithm        |            backbone             |             neck             |                 head                 |                loss                |         memory         |
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

## Construct algorithms from sub-modules

Just as shown in above table, each algorithm is a combination of backbone, neck, head, loss and memories. You are free to use these existing modules to build your own algorithms. If some customized modules are required, you should follow [add_modules](./add_modules.md) to meet your own need.
MMSelfSup provides a base model, called `BaseModel`, and all algorithms
should inherit this base model. And all sub-modules, except for memories, will be built in the base model, during the initialization of each algorithm. Memories will be built in the `__init__` of each specific algorithm. And loss will be built when building the head.

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

Just as shown above, you should provide the config to build the backbone, but neck and head are optional. In addition to building
your algorithm, you should overwrite some abstract functions in the base model to get the correct results, which we will discuss in the
following section.

## Overview these abstract functions in base model

The `forward` function is the entrance to the results. However, it is different from the default `forward` function in most PyTorch code, which
only has one mode. You will mess all your logic in the `forward` function, limiting the scalability. Just as shown in the code below, `forward` function in MMSelfSup has three modes, i) tensor, ii) loss and iii) predict.

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

- tensor, if the mode is `tensor`, the forward function will return the extracted features for images.
  You should overwrite the `extract_feat` to implement your customized extracting process.

- loss, if the mode is `loss`, the forward function will return the loss between the prediction and the target.
  You should overview the `loss` to implement your customized loss function.

- predict, if the mode is `predict`, the forward function will return the prediction, e.g. the predicted label, from
  your algorithm. If should also overwrite the `predict` function.

Now we have introduce the basic components related to models in MMSelfSup, if you want to dive in , please refer the API doc of each algorithm.
