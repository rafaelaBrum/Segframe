#!/usr/bin/env python3
#-*- coding: utf-8

from abc import ABC,abstractmethod
import os

class SegImage(ABC):
    """
    Common abstract class for any image type supported by the system

    @param path <str>: path to image file
    @param keepImg <bool>: keep image data in memory until ordered to release it
    """
    def __init__(self,path,keepImg=False,verbose=0):
        if isinstance(path,str) and os.path.isfile(path):
            self._path = path
        else:
            raise ValueError("[GenericImage] Path does not correspond to a file ({0}).".format(path))

        self._verbose = verbose
        self._keep = keepImg
        self._data = None
        self._dim = None

    def __str__(self):
        """
        String representation is file name
        """
        return os.path.basename(self.path)

    def __eq__(self,other):
        return self._path == other.getPath()

    def __getstate__(self):
          """
          Prepares for pickling.
          """
          state = self.__dict__.copy()
          del state['_data']
          state['_data'] = None

          return state
      
    @abstractmethod
    def __hash__(self):
        pass
    
    @abstractmethod
    def readImage(self,size=None,verbose=None):
        pass

    @abstractmethod
    def readImageRegion(self,x,y,dx,dy):
        pass
    
    @abstractmethod
    def getImgDim(self):
        """
        Should return dimensions as a tuple of (widthd,height,channels)
        """
        pass
    
    def setKeepImg(self,keep):
        """
        If image should not be held anymore, delete data
        """
        if keep is None:
            return
        
        if not keep:
            del self._data
            self._data = None

        self._keep = keep
    
    def getImgName(self):
        return os.path.basename(self._path).split('.')[0]

    def getPath(self):
        return self._path
