import torch
import torch.nn as nn
from packaging import version
from mmcv.cnn import kaiming_init, normal_init

from .registry import NECKS
from .utils import build_norm_layer


def _init_weights(module, init_linear='normal'):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=0.01)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m,
                        (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


@NECKS.register_module
class LinearNeck(nn.Module):
    '''Linear neck: fc only
    '''

    def __init__(self, in_channels, out_channels, with_avg_pool=True):
        super(LinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, out_channels)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.fc(x.view(x.size(0), -1))]


@NECKS.register_module
class NonLinearNeckV0(nn.Module):
    '''The non-linear neck in ODC, fc-bn-relu-dropout-fc-relu
    '''

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV0, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.BatchNorm1d(hid_channels, momentum=0.001, affine=False),
            nn.ReLU(inplace=True), nn.Dropout(),
            nn.Linear(hid_channels, out_channels), nn.ReLU(inplace=True))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module
class NonLinearNeckV1(nn.Module):
    '''The non-linear neck in MoCO v2: fc-relu-fc
    '''
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV1, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module
class NonLinearNeckV2(nn.Module):
    '''The non-linear neck in byol: fc-bn-relu-fc
    '''
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV2, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.BatchNorm1d(hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1, "Got: {}".format(len(x))
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module
class NonLinearNeckSimCLR(nn.Module):
    '''SimCLR non-linear neck.
    Structure: fc(no_bias)-bn(has_bias)-[relu-fc(no_bias)-bn(no_bias)].
        The substructures in [] can be repeated. For the SimCLR default setting,
        the repeat time is 1.
    However, PyTorch does not support to specify (weight=True, bias=False).
        It only support \"affine\" including the weight and bias. Hence, the
        second BatchNorm has bias in this implementation. This is different from
        the offical implementation of SimCLR.
    Since SyncBatchNorm in pytorch<1.4.0 does not support 2D input, the input is
        expanded to 4D with shape: (N,C,1,1). I am not sure if this workaround
        has no bugs. See the pull request here:
        https://github.com/pytorch/pytorch/pull/29626

    Arguments:
        num_layers (int): number of fc layers, it is 2 in the SimCLR default setting.
    '''

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_layers=2,
                 with_avg_pool=True):
        super(NonLinearNeckSimCLR, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse("1.4.0"):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.relu = nn.ReLU(inplace=True)
        self.fc0 = nn.Linear(in_channels, hid_channels, bias=False)
        _, self.bn0 = build_norm_layer(
            dict(type='SyncBN'), hid_channels)

        self.fc_names = []
        self.bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 \
                else hid_channels
            self.add_module(
                "fc{}".format(i),
                nn.Linear(hid_channels, this_channels, bias=False))
            self.add_module(
                "bn{}".format(i),
                build_norm_layer(dict(type='SyncBN'), this_channels)[1])
            self.fc_names.append("fc{}".format(i))
            self.bn_names.append("bn{}".format(i))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = self._forward_syncbn(self.bn0, x)
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            bn = getattr(self, bn_name)
            x = self.relu(x)
            x = fc(x)
            x = self._forward_syncbn(bn, x)
        return [x]


@NECKS.register_module
class AvgPoolNeck(nn.Module):

    def __init__(self):
        super(AvgPoolNeck, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, **kwargs):
        pass

    def forward(self, x):
        assert len(x) == 1
        return [self.avg_pool(x[0])]
