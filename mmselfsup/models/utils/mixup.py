# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value,
                      device=x.device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(
        target,
        num_classes,
        on_value=on_value,
        off_value=off_value,
        device=device)
    y2 = one_hot(
        target.flip(0),
        num_classes,
        on_value=on_value,
        off_value=off_value,
        device=device)
    return y1 * lam + y2 * (1. - lam)


def rand_bbox(img_shape, lam, margin=0., count=None):
    """Standard CutMix bounding-box Generates a random square bbox based on
    lambda value. This impl includes support for enforcing a border margin as
    percent of bbox dimensions.

    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin
        (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape, minmax, count=None):
    """Min-Max CutMix bounding-box Inspired by Darknet cutmix impl, generates a
    random rectangular bbox based on min/max percent values applied to each
    dimension of the input image.

    Typical defaults for minmax are usually in the  .2-.3 for min and
    .8-.9 range for max.

    Args:
        img_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios
            (as percent of image size)
        count (int): Number of bbox to generate
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(
        int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(
        int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cutmix_bbox_and_lam(img_shape,
                        lam,
                        ratio_minmax=None,
                        correct_lam=True,
                        count=None):
    """Generate bbox and apply lambda correction."""
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class Mixup:
    """Mixup/Cutmix that applies different params to each element or whole
    batch.

    Borrow this code from https://github.com/rwightman/pytorch-image-models.

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
            Defaults to 1.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
            Defaults to 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is
            active and uses this vs alpha if not None. Defaults to None.
        prob (float): probability of applying mixup or cutmix
            per batch or element. Defaults to 1.0.
        switch_prob (float): probability of switching to cutmix instead
            of mixup when both are active. Defaults to 0.5.
        mode (str): how to apply mixup/cutmix params
            (per 'batch', 'pair' (pair of elements), 'elem' (element).
            Defaults to batch.
        correct_lam (bool): apply lambda correction when cutmix bbox
            clipped by image borders. Defaults to True.
        label_smoothing (float): apply label smoothing to the
            mixed target tensor. Defaults to 0.1.
        num_classes (int): number of classes for target. Defaults to 1000.
    """

    def __init__(self,
                 mixup_alpha=1.,
                 cutmix_alpha=0.,
                 cutmix_minmax=None,
                 prob=1.0,
                 switch_prob=0.5,
                 mode='batch',
                 correct_lam=True,
                 label_smoothing=0.1,
                 num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam
        self.mixup_enabled = True

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(
                    self.cutmix_alpha,
                    self.cutmix_alpha) if use_cutmix else np.random.beta(
                        self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, 'One of mixup_alpha > 0., cutmix_alpha > 0., \
                    cutmix_minmax not None should be true.'

            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x.shape,
                lam,
                ratio_minmax=self.cutmix_minmax,
                correct_lam=self.correct_lam)
            x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1. - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target):
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        lam = self._mix_batch(x)
        target = mixup_target(target, self.num_classes, lam,
                              self.label_smoothing)
        return x, target


if __name__ == "__main__":

    target = torch.tensor([1, 2])
    import pdb
    pdb.set_trace()
    mixup_loss = mixup_target(target, 10, 0.8, 0.1)
