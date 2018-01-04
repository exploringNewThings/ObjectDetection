import abc
import cv2


class BetaTransforms(metaclass=abc.ABCMeta):
    def __init__(self, name, prob=0.5):
        self._name = name
        self._prob = prob

    def name(self):
        return self._name

    @abc.abstractmethod
    def transform(self, sample):
        pass

    def setprob(self, flip_prob=None):
        assert flip_prob is not None, "Probability value has not been provided."

        assert 0 < flip_prob < 1, "Probability value must be between (excluding) " \
                                  "0 and 1. You provided {}.".format(flip_prob)

        self._prob = flip_prob
        return None