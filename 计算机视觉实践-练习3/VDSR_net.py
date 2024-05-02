import mindspore.nn as nn
from mindspore.common.initializer import Normal
from math import sqrt
from mindspore import Tensor, Parameter
import numpy as np
from mindspore.common.initializer import Normal, initializer
class Conv_ReLU_Block(nn.Cell):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, has_bias=False,pad_mode='pad')
        self.relu = nn.ReLU()

    def construct(self, x):
        return self.relu(self.conv(x))


class VDSRNet(nn.Cell):
    def __init__(self):
        super(VDSRNet, self).__init__()
        self.residual_layer = self.make_layer(Conv_ReLU_Block, 18)
        self.input = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, has_bias=False,pad_mode='pad')
        self.output = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, has_bias=False,pad_mode='pad')
        self.relu = nn.ReLU()

        name_conv= {}

        for _, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                name_conv[_+'.weight'] = n
        for name, param in self.parameters_and_names():
            if name in name_conv:
                n = name_conv[name]
                param.set_data(initializer(Normal(0, sqrt(2. / n)), param.shape, param.dtype))


    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.SequentialCell(layers)

    def construct(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = out + residual
        return out
