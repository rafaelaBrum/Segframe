#!/usr/bin/env python3
#-*- coding: utf-8

from abc import ABC,abstractmethod
import os

class SegImage(ABC):
    """
    Common abstract class for any image type supported by the system
    """
    def __init__(self,path,verbose=0):
        if isinstance(path,str) and os.path.isfile(path):
            self._path = path
        else:
            raise ValueError("[GenericImage] Path does not correspond to a file.")

        self._verbose = verbose


    @abstractmethod
    def readImage(self):
        pass


    @abstractmethod
    def getImgDim(self):
        """
        Should return dimensions as a tuple of (widthd,height)
        """
        pass
