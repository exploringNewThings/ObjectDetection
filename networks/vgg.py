import logging
from .NetworkBase import BaseNetwork
import torch.nn as nn
from utils.timing import timeit

VGG_family = ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16',
              'vgg16_bn',
              'vgg19_bn', 'vgg19', ]

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,
          'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
          512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512,
          512, 'M', 512, 512, 512, 512, 'M'],
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(BaseNetwork):
    def __init__(self, member):
        assert member in VGG_family, "{} is not a valid VGG network type.\n " \
                                     "VGG network should be one of {} or {" \
                                     "}.".format(member, ','.join(VGG_family[
                                                                  :-1]),
                                                 VGG_family[-1])
        logging.debug("The base network has been selected as {}".format(member))
        super(VGG, self).__init__(member)
        logging.debug("Alexnet was successfully defined.")

    @timeit
    def forward(self, data):
        member = self.name()
        logging.debug("Defining {}".format(member))
        if member is 'vgg11':
            return make_layers(cfg['A'])(data)

        if member is 'vgg11_bn':
            return make_layers(cfg['A'], batch_norm=True)(data)

        if member is 'vgg13':
            return make_layers(cfg['B'])(data)

        if member is 'vgg13_bn':
            return make_layers(cfg['B'], batch_norm=True)(data)

        if member is 'vgg16':
            return make_layers(cfg['D'])(data)

        if member is 'vgg16_bn':
            return make_layers(cfg['D'], batch_norm=True)(data)

        if member is 'vgg19':
            return make_layers(cfg['E'])(data)

        if member is 'vgg19_bn':
            return make_layers(cfg['E'], batch_norm=True)(data)
