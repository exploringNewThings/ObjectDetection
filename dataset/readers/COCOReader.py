import os
import logging
import json
import pandas as pd
from .ReaderBase import BaseReader
from utils.timing import timeit


class COCOReader(BaseReader):
    def __init__(self, categories="all"):
        super(COCOReader, self).__init__("MS-COCO Reader", categories)

    def readfile(self, filename, **kwargs):
        assert os.path.isfile(filename), \
            "The annotation file was not found."

        assert (kwargs is not None) or ('imagedir' not in set(kwargs.keys())), \
            "You must pass a single directory (" \
            "imagedirs) where COCO images are stored. " \
            "You have not provided any."

        imagedir = kwargs['imagedir']

        assert os.path.isdir(imagedir), "The  image directory {} " \
                                        "was not found.".format(imagedir)

        df = self._readcocoannotations(filename, imagedir)

        logging.debug("Total size of dataset = {}".format(df.shape[0]))

        return df

    @timeit
    def _readcocoannotations(self, filename, imagedir):
        logging.debug("Opening {}.".format(filename))
        db = json.load(open(filename, 'r'))
        keys = db.keys()
        catdict = {x['name']: x['id'] for x in db['categories']}
        if self._categories != {"all"}:
            assert self._categories.issubset(set(
                catdict.keys())), "Some or all of categories were not found " \
                                  "in the object categories found in the " \
                                  "annotation file {}.".format(filename)

        imgdf = pd.DataFrame(db['images'])
        imgdf.drop(['coco_url', 'date_captured', 'flickr_url', 'license'],
                   axis=1, inplace=True)
        imgdf.set_index('id', inplace=True)

        anndf = pd.DataFrame(db['annotations'])
        anndf.drop(['id', 'segmentation'], axis=1, inplace=True)
        anndf.set_index('image_id', inplace=True)

        combined = anndf.join(imgdf)
        if self._categories != {"all"}:
            categories = [catdict[x] for x in self._categories]
            combined = combined[combined['category_id'].isin(categories)]

        func = lambda x: os.path.join(imagedir, x)

        transformed = combined.transform({'file_name': func})
        combined['file_name'] = transformed['file_name']
        catdictinv = {v: k for k, v in catdict.items()}

        func = lambda x: catdictinv[x]

        catnames = combined.transform({'category_id': func})
        combined['category_name'] = catnames

        func = lambda x: list(map(int, x))
        bbox = combined.transform({'bbox': func})
        combined['bbox'] = bbox

        return combined

    def savepath(self, imagepath):
        assert self._basesavepath is not None, "No base folder for saving " \
                                               "detections is known. Please " \
                                               "use setsavepath() for setting " \
                                               "the base folder."
        imagefilename = os.path.basename(imagepath)
        savepath = os.path.join(self._basesavepath, imagefilename)
        return savepath
