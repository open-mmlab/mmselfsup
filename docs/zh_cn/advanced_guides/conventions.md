# 约定

- [约定](#约定)
  - [损失](#损失)

如果您想将 MMSelfSup 修改为您自己的项目, 请检查以下约定。

## 损失

当算法实现时, 函数 `loss` 返回的损失应该是 `dict` 类型。

举个 `MAE` 的例子：

```python
class MAE(BaseModel):
    """MAE.

    Implementation of `Masked Autoencoders Are Scalable Vision Learners
    <https://arxiv.org/abs/2111.06377>`_.
    """

    def extract_feat(self, inputs: List[torch.Tensor],
                     **kwarg) -> Tuple[torch.Tensor]:
        ...

    def loss(self, inputs: List[torch.Tensor],
             data_samples: List[SelfSupDataSample],
             **kwargs) -> Dict[str, torch.Tensor]:
        """The forward function in training.

        Args:
            inputs (List[torch.Tensor]): The input images.
            data_samples (List[SelfSupDataSample]): All elements required
                during the forward function.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of loss components.
        """
        # ids_restore: the same as that in original repo, which is used
        # to recover the original order of tokens in decoder.
        latent, mask, ids_restore = self.backbone(inputs[0])
        pred = self.neck(latent, ids_restore)
        loss = self.head(pred, inputs[0], mask)
        losses = dict(loss=loss)
        return losses

```

在 `MAE` 模型正向传播期间, 这个 `MAE.loss()` 函数将被调用用于计算损失并返回这个损失值。

默认情况下, 只有 `dict` 中的键包含的 `loss` 值时, 才会进行反向传播, 如果你的算法需要多个损失值, 你可以用多个键打包损失字典。

```python
class YourAlgorithm(BaseModel):

    def loss():
        ...

        losses['loss_1'] = loss_1
        losses['loss_2'] = loss_2
```
