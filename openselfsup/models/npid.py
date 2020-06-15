import torch
import torch.nn as nn

from openselfsup.utils import print_log

from . import builder
from .registry import MODELS


@MODELS.register_module
class NPID(nn.Module):
    '''Model of "Unsupervised Feature Learning via Non-parametric
       Instance Discrimination".
    Arguments:
        neg_num (int): number of negative samples for each image
        ensure_neg (bool): if False, there is a small probability
            that negative samples contain positive ones.
    '''

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 memory_bank=None,
                 neg_num=65536,
                 ensure_neg=False,
                 pretrained=None):
        super(NPID, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.memory_bank = builder.build_memory(memory_bank)
        self.init_weights(pretrained=pretrained)

        self.neg_num = neg_num
        self.ensure_neg = ensure_neg

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            print_log('load model from: {}'.format(pretrained), logger='root')
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights(init_linear='kaiming')

    def forward_backbone(self, img):
        """Forward backbone

        Returns:
            x (tuple): backbone outputs
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, idx, **kwargs):
        x = self.forward_backbone(img)
        idx = idx.cuda()
        feature = self.neck(x)[0]
        feature = nn.functional.normalize(feature)  # BxC
        bs, feat_dim = feature.shape[:2]
        neg_idx = self.memory_bank.multinomial.draw(bs * self.neg_num)
        if self.ensure_neg:
            neg_idx = neg_idx.view(bs, -1)
            while True:
                wrong = (neg_idx == idx.view(-1, 1))
                if wrong.sum().item() > 0:
                    neg_idx[wrong] = self.memory_bank.multinomial.draw(
                        wrong.sum().item())
                else:
                    break
            neg_idx = neg_idx.flatten()

        pos_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      idx)  # BXC
        neg_feat = torch.index_select(self.memory_bank.feature_bank, 0,
                                      neg_idx).view(bs, self.neg_num,
                                                    feat_dim)  # BxKxC

        pos_logits = torch.einsum('nc,nc->n',
                                  [pos_feat, feature]).unsqueeze(-1)
        neg_logits = torch.bmm(neg_feat, feature.unsqueeze(2)).squeeze(2)

        losses = self.head(pos_logits, neg_logits)

        # update memory bank
        with torch.no_grad():
            self.memory_bank.update(idx, feature.detach())

        return losses

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.forward_backbone(img)
        else:
            raise Exception("No such mode: {}".format(mode))
