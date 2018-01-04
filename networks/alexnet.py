import logging
from .NetworkBase import BaseNetwork
import torch.nn as nn
from utils.timing import  timeit


class AlexNet(BaseNetwork):
    def __init__(self):
        logging.debug("The base network has been selected as Alexnet.")
        super(AlexNet,self).__init__("AlexNet")


    @timeit
    def forward(self, data):
        logging.debug("Defining the network.")
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(1,11), stride=1, padding=(0,5)),
            nn.Conv2d(64, 64, kernel_size=(11,1), stride=1, padding=(5,0)),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.MaxPool2d(kernel_size=3, stride=2),
        )

        return model(data)
