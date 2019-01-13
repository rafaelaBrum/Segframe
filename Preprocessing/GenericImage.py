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
    def __init__(self,path,verbose=0):
        """
        @param path <str>: path to image
        """
        super().__init__(path,verbose)

    def readImage(self):
        pass

    def getImgDim(self):
        pass
