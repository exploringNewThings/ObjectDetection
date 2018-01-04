import abc
import os
import logging


class BaseReader(metaclass=abc.ABCMeta):
    """An Abstract Base Class(ABC) for constructing custom readers for use
    with the system. """
    def __init__(self, readertype, categories= "all"):
        """
        Initializer for the class BaseReader. When initializing a
        class, subclassing from BaseReader, an implementer must ensure that
        BaseReader is explicitly initialized.
        Args:
            readertype (str): A name given to the reader. Ideally it should
                       describe the kind of the reader.
            categories (str or list(str)): Name(s) of the object categories
                       whose annotations should be read by the reader. A
                       special value of "all" ensures that annotations for all
                       the object categories are read. Defaults to "all".

        """
        assert readertype is not None, "Every reader must have a name."
        self._readertype = readertype
        self._categories = {categories} if \
            isinstance(categories,str) else set(categories)

        self._basesavepath = None

    def readertype(self):
        """
        Returns the name of the reader.
        Returns:
            str : Name of the reader.

        """
        return self._readertype

    def setsavepath(self, basesavepath):
        if not os.path.exists(basesavepath):
            logging.debug(
                "The specified basepath {} does not exist. Creating it.".format(
                    basesavepath))
            try:
                os.makedirs(basesavepath)
            except:
                logging.FATAL("The basepath {} could not be created".format(
                    basesavepath))

        self._basesavepath = basesavepath
        return None

    @abc.abstractmethod
    def readfile(self, filenames, **kwargs):
        """
        An Abstract Method. It serves the purpose of reading (possibly
        multiple !) annotation files and coalescing them into a
        `pandas.DataFrame(). <https://pandas.pydata.org/pandas-docs/stable
        /generated/pandas.DataFrame.html>`_ The coalesced DataFrame() is then returned.
        Args:
            filenames (str or list(str)): Name(s) of annotation file(s) which
                   have to be read by the reader.

            **kwargs: Other arguments. Depend upon the specific details of
                   concrete child classes

        Returns :
            pandas.DataFrame : A DataFrame() with at least two columns (
            'file_name' and 'bbox'). 'file_name' should encode the full path
            to the image(s) and 'bbox' should encode the bounding box
            coordinates of the rectangle(s) in the format [Top-Left-Row,
            Top-Left-Column, Width, Height]. Each row encodes information for ONLY one object.

        """
        pass

    @abc.abstractmethod
    def savepath(self, imagepath):
        """
        An Abstract Method. To each image in a dataset, it assigns a
        corresponding full path (directory + filename), where the user would
        like to save the bounding box detections. Different datasets are
        organized differently. While in datasets such as MSCOCO and Pascal
        VOC, all images lie at the same directory level, in other datasets
        like Caltech Pedestrian Dataset and ILSVRC, the structure is more
        complex. This abstract method has been specially created to
        facilitate the saving of detections for arbitrary datasets.
        Args:
            imagepath (str): The full path to an image file.

        Returns:
             (str) : The full path for the image with visualized bounding box
             detection(s).

        """
        pass

    def addsaveinfo(self, df):
        func = lambda x : self.savepath(x)
        saveinfo = df.transform({'file_name' : func})
        df = df.assign(save_name = saveinfo)
        return df


