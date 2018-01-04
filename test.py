import logging
import numpy as np
from dataset.readers import COCOReader
#from networks.alexnet import AlexNet 
#from networks.vgg import VGG 
#from utils import plotting 
from dataset import InputDataset
from transforms import individualfliphorizontal, imagefliphorizontal, brightness
FORMAT = '%(asctime)s : %(pathname)s : %(module)s : %(message)s'
logging.basicConfig(level=logging.DEBUG, filename="test.log", filemode='w',
format=FORMAT)

reader = COCOReader.COCOReader(categories='person')
reader.setsavepath('/data/stars/user/uujjwal/mscocogt')
df = reader.readfile(
"/data/stars/user/uujjwal/datasets/generic-detection/mscoco/annotations/instances_train2014.json"
,
imagedir="/data/stars/user/uujjwal/datasets/generic-detection/mscoco/train2014")
df = reader.addsaveinfo(df)
print(df.shape)
print(df.head)


fliptransform = imagefliphorizontal.ImageFlipHorizontal(prob=0.99)
brighttransform = brightness.ObjectBrightness(prob=0.999, random_factor = 0.6)
db = InputDataset.InputDataset(df, transforms=[fliptransform, brighttransform])
for i in range(10):
    choice = np.random.choice(range(len(db._uniquefiles)))
    sample = db.__getitem__(choice)
    print(sample)

