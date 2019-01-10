#!/usr/bin/env python3
#-*- coding: utf-8

from abc import ABC,abstractmethod
import os

class ImageSource(ABC):
    """
    Abstract class for image sources. Provides a comprehensive list of supported images in a dataset.
    """
    def __init__(self,path,img_type,verbose=0):
        if not os.path.isdir(path):
            raise ValueError("[GenericData] Path should point to an input image dir: {0}".format(path))

        self.path = path
        self._verbose = verbose
        self.img_type = img_type

    @abstractmethod
    def getImgList(self):
        pass
