import numpy as np
import cv2

from tests import InputDataset_test as InputDataset
from dataset.readers import COCOReader
from transforms.normalize import Normalize

def main():
    reader = COCOReader.COCOReader(categories='person')
    df = reader.readfile("/data/stars/user/uujjwal/datasets/generic-detection/mscoco/annotations/instances_train2014.json", imagedir="/data/stars/user/uujjwal/datasets/generic-detection/mscoco/train2014")

    reader.setsavepath('/data/stars/user/agoel/tmp')
    
    df = reader.addsaveinfo(df)

    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    train_dataset = InputDataset.InputDataset(df, transforms=[normalize])

    sample = train_dataset.__getitem__(0)
    
    print(sample)

if __name__ == '__main__':
    main()
