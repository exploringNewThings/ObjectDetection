import random
import numpy as np
import cv2
from .transforms import BetaTransforms


class ObjectBrightness(BetaTransforms):
    def __init__(self, prob=0.5, random_factor=0.2):
        super(ObjectBrightness, self).__init__("Random Brightness Transform",
                                             prob)
        assert 0<random_factor<1, "The random factor must be a float between " \
                                  "0 and 1."
        self._random_factor = random_factor

    def transform(self, sample):
        img = sample['image']
        for ann in sample['annotations']:
            gt = ann[1]
            modify = np.random.choice([True, False],
                                    p=[self._prob, 1 - self._prob])
            if modify:
                patch = img[gt[1]: gt[1] + gt[3] + 1, gt[0]: gt[0] + gt[2] + 1]
                meanbrightness = np.mean(patch)
                randomshift = random.uniform(-self._random_factor,
                                             self._random_factor)
                change = int(randomshift * meanbrightness)
                newpatch = np.zeros(patch.shape, dtype='uint8')
                patch = cv2.convertScaleAbs(patch, newpatch, 1, change)
                img[gt[1]: gt[1] + gt[3] + 1, gt[0]: gt[0] + gt[2] + 1] = \
                    newpatch

        sample['image'] = img
        return sample
