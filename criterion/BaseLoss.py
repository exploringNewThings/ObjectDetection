import logging
import numpy as np
import cv2
import torch
from torch.autograd import Variable
'''
def BaseLoss(feature_maps=None, bboxes = None, orig_size=None):
    assert feature_maps is not None,logging.FATAL("No feature maps have been provided.")

    assert bboxes is not None, logging.FATAL("No bounding boxes have been provided.")

    assert orig_size is not None, logging.FATAL("Original image size has not been provided.")

    #Here orig_size is assumed to be of the type 3 x width x height
    batchsize = feature_maps.shape[0]

    losses = []

    for batch in range(batchsize):
        feat2 = feature_maps[batch,:,:,:].cpu().detach().numpy()
        feat2 = np.transpose(feat2,(2,1,0))
        #print(feat2.shape)
        #print(orig_size[1],orig_size[2])
        feat = np.zeros((orig_size[1], orig_size[2], feat2.shape[2]), dtype=np.float32)
        
        for i in range(feat2.shape[2]):
            feat[:,:,i] = cv2.resize(feat2[:,:,i],(orig_size[2], orig_size[1]), cv2.INTER_CUBIC)
        #at = scipy.misc.imresize(feat,(orig_size[1],orig_size[2]))
        feat2 = np.transpose(feat2,(2,1,0))
        mask,bbox_pixels,non_bbox_pixels = create_mask(feat.shape[1:], bboxes[batch])
        feat = np.square(feat)
        feat = np.sum(feat, axis=0)
        bboxvalues = float(np.sum(feat[mask])/(bbox_pixels*feat.shape[0]))
        restvalues = float(np.sum(feat[~mask])/(non_bbox_pixels*feat.shape[0]))
        losses.append(restvalues - bboxvalues)
        losses = Variable(torch.DoubleTensor(losses),requires_grad=True)

    return sum(losses)/len(losses),feat
'''
def BaseLoss(feature_maps=None, bboxes = None, orig_size=None):
    assert feature_maps is not None,logging.FATAL("No feature maps have been provided.")

    assert bboxes is not None, logging.FATAL("No bounding boxes have been provided.")

    assert orig_size is not None, logging.FATAL("Original image size has not been provided.")

    #Here orig_size is assumed to be of the type 3 x width x height                                      
    batchsize = feature_maps.shape[0]
    losses = []

    #print(orig_size)
    H = orig_size[0].numpy()
    W = orig_size[1].numpy()
    C = orig_size[2].numpy()
    #print(len(orig_size))
    #print(np.shape(bboxes))
    print(bboxes)
    bboxes = bboxes[0]
    bboxes = bboxes[1]
    #print(bboxes)
    for batch in range(batchsize):
        feat2 = feature_maps[batch,:,:,:].cpu().detach().numpy()
        feat2 = np.transpose(feat2,(2,1,0))
        #print(feat2.shape)  
        #print(" W = {}".format(W[batch]))
        #print(" H = {}".format(H[batch]))

        feat2 = cv2.resize(feat2,(W[batch],H[batch]), cv2.INTER_CUBIC)
        
        #at = scipy.misc.imresize(feat,(orig_size[1],orig_size[2]))                                      
        feat2 = np.transpose(feat2,(2,1,0))
        mask,bbox_pixels,non_bbox_pixels = create_mask(feat2.shape[1:], bboxes[batch])
        feat = np.square(feat2)
        feat = np.sum(feat, axis=0)
        bboxvalues = float(np.sum(feat[mask])/(bbox_pixels*feat.shape[0]))
        restvalues = float(np.sum(feat[~mask])/(non_bbox_pixels*feat.shape[0]))
        losses.append(restvalues - bboxvalues)
        losses = Variable(torch.DoubleTensor(losses),requires_grad=True)

    return sum(losses)/len(losses),feat

def create_mask(mask_size, bboxes):
    mask = np.zeros(mask_size, dtype=bool)
    bbox_pixels = 0
    non_bbox_pixels = 0
    #print('----------------------------')
    #print('BBOXES IS {}'.format(bboxes))
    #print(np.shape(bboxes))
    #print(len(bboxes))
    #print(bboxes)
    for bb in bboxes:#np.asarray(bboxes[1:],dtype='int32'):
        #print('The value of bb is {}'.format(bb))
        mask[bb[1] : bb[1] + bb[3] - 1, bb[0] : bb[0] + bb[2] - 1] = True
        bbox_pixels += (bb[2]*bb[3])
    
    non_bbox_pixels += ((mask_size[0]*mask_size[1])-bbox_pixels)

    return mask,bbox_pixels,non_bbox_pixels


