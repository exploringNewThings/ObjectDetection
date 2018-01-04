import abc
import logging
from collections import OrderedDict
import torch
from torch.autograd import Variable
import pandas as pd
from utils.timing import timeit


class BaseNetwork(metaclass=abc.ABCMeta):
    def __init__(self, name):
        self._name = name

    @abc.abstractmethod
    def forward(self, data):
        pass

    def name(self):
        return self._name

