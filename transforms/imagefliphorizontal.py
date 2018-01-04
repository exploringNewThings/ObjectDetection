import numpy as np
import cv2
from .transforms import BetaTransforms


class ImageFlipHorizontal(BetaTransforms):
    def __init__(self, prob=0.5):
        super(ImageFlipHorizontal, self).__init__("Image level Horizontal "
                                                  "Flip",prob)

    def transform(self, sample):
        flip = np.random.choice([True, False], p=[self._prob, 1 - self._prob])
        if flip:
            img = sample['image']
            img = cv2.flip(img, flipCode=1)
            annotations = sample['annotations']
            width = img.shape[1]
            for ind in range(len(annotations)):
                gt = annotations[ind][1]
                gt[0] = width - gt[0] - gt[2] - 1
                annotations[ind][1] = gt
            sample['annotations'] = annotations
            sample['image'] = img

        return sample

