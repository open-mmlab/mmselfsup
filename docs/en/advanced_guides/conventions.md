# Conventions

Please check the following conventions if you would like to modify MMSelfSup as your own project.

## Losses

When the algorithm is implemented, the returned losses is supposed to be `dict` type.

Take `MAE` as an example:

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

The `MAE.loss()` function will be called during model forward to compute the loss and return its value.

By default, only values whose keys contain `'loss'` will be back propagated, if your algorithm need more than one loss value, you could pack losses dict with several keys:

```python
class YourAlgorithm(BaseModel):

    def loss():
        ...

        losses['loss_1'] = loss_1
        losses['loss_2'] = loss_2
```
