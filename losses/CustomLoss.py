import torch
import torch.nn as nn
from criterion.BaseLoss import BaseLoss

class BoundingBoxLoss(nn.Module):
    '''Loss computed based on the difference in energy between the region in the feature map 
    that is not having the bounding box and the one that is having.
    '''
    def __init__(self):
        super(BoundingBoxLoss, self).__init__()

    def forward(self, feature_maps=None, bboxes = None, orig_size=None):
        loss = BaseLoss(feature_maps, bboxes, orig_size)
        return loss
