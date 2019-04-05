#!/usr/bin/env python3
#-*- coding: utf-8

import os
import imghdr
from .ImageSource import ImageSource
from Preprocessing import GenericImage

class GenericData(ImageSource):
    """
    Used for datasets that don't have a predefined structure.
    """
    def __init__(self,path,img_type,verbose=0):
        """
        @param path <str>: Path do data directory
        @param img_type <list>: list of supported image types
        """
        super().__init__(path,img_type,verbose)
        self._data = None
        self._run()

    def _run(self):
        """
        Reads dir and builds the data dir with information about images found
        """
        def _checkType(impath):
            if not self.img_type is None:
                fields = impath.split('.')

                #Even if file has no extension, check to see if its an image of given type. This may be slow!!
                if len(fields) == 1 and imghdr.what(os.path.join(self.path,impath)) in self.img_type:
                    return impath
                
                if fields[1] in self.img_type:
                    return impath
                
                return None
            else:
                return impath

        self._data = list(filter(_checkType,os.listdir(self.path)))

    def getImgList(self):
        """
        Returns a list of GenericImage objects (ones tractable by OpenCV).
        """
        return [GenericImage(p) for p in self._data]
        

