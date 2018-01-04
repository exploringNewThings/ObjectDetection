import numpy as np
import cv2
from .transforms import BetaTransforms


class IndividualFlipHorizontal(BetaTransforms):
    def __init__(self, prob=0.5):
        super(IndividualFlipHorizontal, self).__init__("Object level "
                                                       "Horizontal Flip",
                                                       prob)

    def transform(self, sample):
        img = sample['image']
        for ann in sample['annotations']:
            gt = ann[1]
            flip = np.random.choice([True, False],
                                    p=[self._prob, 1 - self._prob])
            if flip:
                img[gt[1]: gt[1] + gt[3] + 1, gt[0]: gt[0] + gt[2] + 1] = \
                    cv2.flip(img[gt[1]: gt[1] + gt[3] + 1, gt[0]: gt[0] + gt[
                        2] + 1], flipCode=1)
                sample['image'] = img

        return sample
