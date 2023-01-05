# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.model import BaseModule

from mmselfsup.registry import MODELS


@MODELS.register_module()
class MAEPretrainHead(BaseModule):
    """Pre-training head for MAE.

    Args:
        loss (dict): Config of loss.
        norm_pix_loss (bool): Whether or not normalize target.
            Defaults to False.
        patch_size (int): Patch size. Defaults to 16.
    """

<<<<<<< HEAD
    def __init__(self,
                 loss: dict,
                 norm_pix: bool = False,
                 patch_size: int = 16) -> None:
=======
    def __init__(self, norm_pix: bool = False, patch_size: int = 16) -> None:
>>>>>>> upstream/master
        super().__init__()
        self.norm_pix = norm_pix
        self.patch_size = patch_size
        self.loss = MODELS.build(loss)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
<<<<<<< HEAD
        """Split images into non-overlapped patches.

        Args:
            imgs (torch.Tensor): A batch of images, of shape B x H x W x C.

        Returns:
            torch.Tensor: Patchified images. The shape is B x L x D.
=======
        """
        Args:
            imgs (torch.Tensor): The shape is (N, 3, H, W)
        Returns:
            x (torch.Tensor): The shape is (N, L, patch_size**2 *3)
>>>>>>> upstream/master
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
<<<<<<< HEAD
        """Combine non-overlapped patches into images.

=======
        """
>>>>>>> upstream/master
        Args:
            x (torch.Tensor): The shape is (N, L, patch_size**2 *3)
        Returns:
            imgs (torch.Tensor): The shape is (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

<<<<<<< HEAD
    def construct_target(self, target: torch.Tensor) -> torch.Tensor:
        """Construct the reconstruction target.

        In addition to splitting images into tokens, this module will also
        normalize the image according to ``norm_pix``.

        Args:
            target (torch.Tensor): Image with the shape of B x 3 x H x W

        Returns:
            torch.Tensor: Tokenized images with the shape of B x L x C
        """
        target = self.patchify(target)
=======
    def forward(self, x: torch.Tensor, pred: torch.Tensor,
                mask: torch.Tensor) -> dict:
        losses = dict()
        target = self.patchify(x)
>>>>>>> upstream/master
        if self.norm_pix:
            # normalize the target image
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        return target

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Forward function of MAE head.

        Args:
            pred (torch.Tensor): The reconstructed image.
            target (torch.Tensor): The target image.
            mask (torch.Tensor): The mask of the target image.

        Returns:
            torch.Tensor: The reconstruction loss.
        """
        target = self.construct_target(target)
        loss = self.loss(pred, target, mask)

<<<<<<< HEAD
        return loss
=======
    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
    """

    def __init__(self, embed_dim, num_classes=1000, label_smooth_val=0.1):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)
        self.criterion = LabelSmoothLoss(label_smooth_val, num_classes)

    def init_weights(self):
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=2e-5)

    def forward(self, x):
        """"Get the logits."""
        outputs = self.head(x)

        return [outputs]

    def loss(self, outputs, labels):
        """Compute the loss."""
        losses = dict()
        losses['loss'] = self.criterion(outputs[0], labels)

        return losses


@HEADS.register_module()
class MAELinprobeHead(BaseModule):
    """Linear probing head for MAE.

    Args:
        embed_dim (int): The dim of the feature before the classifier head.
        num_classes (int): The total classes. Defaults to 1000.
    """

    def __init__(self, embed_dim, num_classes=1000):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)
        self.bn = nn.BatchNorm1d(embed_dim, affine=False, eps=1e-6)
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self):
        nn.init.constant_(self.head.bias, 0)
        trunc_normal_(self.head.weight, std=0.01)

    def forward(self, x):
        """"Get the logits."""
        x = self.bn(x)
        outputs = self.head(x)

        return [outputs]

    def loss(self, outputs, labels):
        """Compute the loss."""
        losses = dict()
        losses['loss'] = self.criterion(outputs[0], labels)

        return losses
>>>>>>> upstream/master
