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
        self._idx = 0
        if isinstance(transforms, BetaTransforms):
            self._transforms = [transforms]
        else:
            self._transforms = transforms if transforms is not None else None

    def __len__(self):
        return len(self._uniquefiles)

    def _readannotations(self,df):
        return list(map(self._gtfunc, df['category_name'], df['bbox']))

    def __getitem__(self, idx):
        idx = self._idx
        batch_size = 16
        total_files = self.__len__()
        
        if (idx+batch_size) > total_files:
            num_files = total_files-idx
        else:
            num_files = batch_size
        
        filename = self._uniquefiles[idx:idx+num_files]
        #filename = filename[0]
        max_h = 0
        max_w = 0
        for x in range(num_files):
            img = cv2.imread(filename[x])
            h,w,c = img.shape
            if h>max_h:
                max_h = h
            if w>max_w:
                max_w = w
        
        image_batch = []
        image_size = []
        tempdf_batch = []
        savename_batch = []
        annotations_batch =[]
        for x in range(num_files):
            img = cv2.imread(filename[x])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h,w,c = img.shape
            pad_h = max_h-h+1
            pad_w = max_w-w+1
            img = cv2.copyMakeBorder(img,0,pad_h,0,pad_w,cv2.BORDER_CONSTANT,0)
            tempdf = self._dataframe.loc[self._dataframe['file_name'] == filename[x]]
            annotations = self._readannotations(tempdf)
            image_batch.append(np.transpose(img,(2,1,0)))
            image_size.append(img.shape)
            tempdf_batch.append(tempdf)
            savename_batch.append(tempdf['save_name'].iloc[0])
            annotations_batch.append(annotations)

        #image = cv2.imread(filename)
        #h,w,c = image.shape
        #image = cv2.resize(image,(299,299),cv2.INTER_CUBIC)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #tempdf = self._dataframe.loc[self._dataframe['file_name'] == filename]
        #savename = tempdf['save_name'].iloc[0]
        #annotations = self._readannotations(tempdf)
        sample = {'image' : image_batch, 'annotations' : annotations_batch, 'file_name':
                  filename, 'save_name': savename_batch, 'orig_size':image_size}

        if self._transforms is not None:
            shuffle(self._transforms)
            smpl = {}
            for x in range(num_files):
                smpl['image'] = image_batch[x]
                smpl['annotations'] = annotations_batch[x]
                smpl['file_name'] = filename[x]
                smpl['save_name'] = savename_batch[x]
                smpl['orig_size'] = image_size[x]
                self.applytransforms(float(smpl))
                sample['image'][x] = smpl['image']
                sample['annotations'][x] = smpl['annotations']
                sample['file_name'][x] = smpl['file_name']
                sample['save_name'][x] = smpl['save_name']
                sample['orig_size'][x] = smpl['orig_size']
            #self.applytransforms(sample)

        #sample['image'] = np.transpose(sample['image'],(2,1,0))
        self._idx = idx+num_files
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



