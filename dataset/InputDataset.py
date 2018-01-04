import os
import logging
from random import shuffle
import numpy as np
import cv2
import pandas as pd
import torch as pt
import torch.utils.data.dataset as D
import torch.utils.data.dataset
from utils.plotting import plotonimage
from torch.autograd import Variable as V
from transforms.transforms import BetaTransforms



class InputDataset(D.Dataset):

    def __init__(self, dataframe, transforms = None):
        self._dataframe = dataframe
        self._uniquefiles = self._dataframe['file_name'].unique().tolist()
        self._gtfunc = lambda x,y : [x,y]
        if isinstance(transforms, BetaTransforms):
            self._transforms = [transforms]
        else:
            self._transforms = transforms if transforms is not None else None

    def __len__(self):
        return len(self._uniquefiles)

    def _readannotations(self,df):
        return list(map(self._gtfunc, df['category_name'], df['bbox']))

    def __getitem__(self, idx):
        filename = self._uniquefiles[idx]
        image = cv2.imread(filename)
        h,w,c = image.shape
        image = cv2.resize(image,(299,299),cv2.INTER_CUBIC)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tempdf = self._dataframe.loc[self._dataframe['file_name'] == filename]
        savename = tempdf['save_name'].iloc[0]
        annotations = self._readannotations(tempdf)
        print(annotations)
        sample = {'image' : image, 'annotations' : annotations, 'file_name':
                  filename, 'save_name': savename, 'orig_size':[h,w,c]}

        if self._transforms is not None:
            shuffle(self._transforms)
            self.applytransforms(sample)

        sample['image'] = np.transpose(sample['image'],(2,1,0))
        #sample['annotations'] = torch.FloatTensor(sample['annotations'])
        #sample['image'] = V(torch.FloatTensor(sample['image']), requires_grad=False)

        return sample

    def plotbb(self, sample):
        img = sample['image'].copy()
        df = pd.DataFrame(sample['annotations'], columns=['category_name',
                                                          'bbox'])
        savename = os.path.join(os.path.dirname(sample['save_name']),
                                'transformed', os.path.basename(sample[
                                                                    'save_name']))

        if not os.path.isdir(os.path.dirname(savename)):
            logging.debug("The path to save the transformed image is not "
                          "available. Creating it.")
            os.makedirs(os.path.dirname(savename))
            logging.debug("The path {} was created")

        plotonimage(img, df, savename)

        return None

    def applytransforms(self, sample):
        for item in self._transforms:
            sample = item.transform(sample)

        return sample



