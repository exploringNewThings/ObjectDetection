import argparse
import logging
import os
import torch
import torch.distributed as dist
from networks.inception import inception_v3
from networks.inception import Inception3Conv
import numpy as np
import logging
#import dataset.readers import COCOReader
#from transforms import individualfliphorizontal, imagefliphorizontal, brightness

def main():

model = inception_v3(pretrained=True)
model.cuda()

randdata = torch.autograd.Variable(torch.randn(1,3,224,224))

out = model.forward(randdata)

print(out)
#model = Inception3Conv(original_model,aux_logits=False)

#randdata = torch.autograd.Variable(torch.randn(1,3,299,299))

#out = model.forward(randdata)
#conv_output = original_model.forward(randdata,onlyConv=True)

#reader = COCOReader.COCOReader(categories='person')

#df = reader.readfile(
#    "/data/stars/user/uujjwal/datasets/generic-detection/mscoco/annotations/instances_train2014.json"
#    ,
#    imagedir="/data/stars/user/uujjwal/datasets/generic-detection/mscoco/train2014")

#print(df.shape)

#reader.setsavepath('/data/stars/user/uujjwal/mscocogt')

#df = reader.addsaveinfo(df)

#fliptransform = imagefliphorizontal.ImageFlipHorizontal(prob=0.99)
#brighttransform = brightness.ObjectBrightness(prob=0.999, random_factor = 0.6)
#db = InputDataset.InputDataset(df, transforms=[fliptransform, brighttransform])

if __name__ == '__main__':
    main()
