import os
from .ReaderBase import BaseReader
from bs4 import BeautifulSoup


class VOCReader(BaseReader):
    def __init__(self, context):
        super(BaseReader, self).__init__("Pascal VOC Reader", context)

    def readfile(self, filenames, categories):
        assert all(list(map(lambda x: os.path.isfile(x), filenames))), \
            "One or more annotation file(s) were not found."



