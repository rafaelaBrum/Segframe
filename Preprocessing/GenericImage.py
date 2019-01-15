#!/usr/bin/env python3
#-*- coding: utf-8

import cv2
import os
import numpy as np

from .SegImage import SegImage

class GenericImage(SegImage):
    """
    Represents any image handled by OpenCV.
    """
    def __init__(self,path,keepImg=False,verbose=0):
        """
        @param path <str>: path to image
        @param keepImg <bool>: keep image data in memory
        """
        super().__init__(path,keepImg,verbose)

    def readImage(self):
        data = cv2.imread(self._path)
        if self._keep:
            self._data = data

        return data
    
    def readImageRegion(self,x,y,dx,dy):
        data = None
        
        if self._data is None:
            data = self.readImage()
        else:
            data = self._data
            
        return data[y:(y+dy), x:(x+dx)]

    def setKeepImg(self,keep):
        """
        If image should not be held anymore, delete data
        """
        if not keep:
            del self._data
            self._data = None

        self._keep = keep
        
    def getImgDim(self):
        pass
