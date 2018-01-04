import abc
import logging

class LossBase(metaclass=abc.ABCMeta):
    def __init__(self, name):
        self._name = name

    @abc.abstractmethod
    def loss(self,*args, **kwargs):
        pass