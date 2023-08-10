import logging

import torch.nn as nn
from torchvision.models import ResNet
from torchvision.models.resnet import Bottleneck

from deepsleep.models import BaseModel
from deepsleep import ROOT_LOGGER_STR

logger = logging.getLogger(ROOT_LOGGER_STR + '.' + __name__)


class SleepResNet3(BaseModel):
    def __init__(self, params):
        super().__init__(params, num_classes=3)

    def set_graph(self, example_set):
        # Set up the model
        resnet = ResNet(Bottleneck, [3, 4, 23, 3],
                        num_classes=self.num_classes)
        resnet.conv1 = nn.Conv2d(example_set.input_dim[1], 64,
                                 kernel_size=7, stride=2,
                                 padding=3, bias=False)
        resnet.float().to(self.device)
        self.graph = nn.Sequential(resnet, nn.LogSoftmax(dim=1))
        self.graph.float().to(self.device)

        pars = sum(p.numel() for p in self.graph.parameters()
                   if p.requires_grad)
        logger.debug(f'Graph created and put to the device {self.device_name}')
        logger.debug(f'{pars} trainable parameters')

