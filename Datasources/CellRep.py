#!/usr/bin/env python3
#-*- coding: utf-8

import numpy as np
import os
import random

#Local modules
from Datasources import GenericDatasource as gd
from Preprocessing import PImage
from Utils import CacheManager

class CellRep(gd.GenericDS):
    """
    Class that parses label.txt text files and loads all images into memory
    """

    def __init__(self,data_path,keepImg=False,config=None):
        """
        @param data_path <str>: path to directory where image patches are stored
        @param config <argparse>: configuration object
        @param keepImg <boolean>: keep image data in memory
        """
        super().__init__(data_path,keepImg,config)
        self.nclasses = 2


    def _load_metadata_from_dir(self,d):
        """
        Create SegImages from a directory
        """
        class_set = set()
        labels = open(os.path.join(d,'label.txt'),'r')

        t_x,t_y = ([],[])
        for f in labels:
            tmp = f.strip().split()
            f_name,f_label = tmp[0],tmp[1]
            origin=''
            coord=None
            if len(tmp) > 2:
                origin = tmp[2]
            if len(tmp) > 4:
                coord = (tmp[3],tmp[4])
            t_path = os.path.join(d,f_name)
            if os.path.isfile(t_path):
                seg = PImage(t_path,keepImg=self._keep,origin=origin,coord=coord,verbose=self._verbose)
                t_x.append(seg)
                t_y.append(int(f_label))
                class_set.add(f_label)
            elif self._verbose > 0:
                print("Label file contains reference to {0}, but no such file exists.".format(t_path))

        #Non-lymphocyte patches are labeld 0 or -1 (no lymphocyte or below lymphocyte threshold)
        # -1 and 0 labels are treated as the same as for now this is a binary classification problem
        if self._verbose > 1:
            print("On directory {2}:\n - Number of classes: {0};\n - Classes: {1}".format(len(class_set),class_set,os.path.basename(d)))

        return t_x,t_y

    def get_dataset_dimensions(self):
        """
        Returns the dimensions of the images in the dataset. It's possible to have different image dimensions.
        WARNING: big datasets will take forever to run. For now, checks a sample of the images.
        TODO: Reimplement this function to be fully parallel (threads in case).

        Return: SORTED list of tuples (# samples,width,height,channels)
        """

        dims = set()
        samples = len(self.X)

        cache_m = CacheManager()
        if cache_m.checkFileExistence('data_dims.pik'):
            dims = cache_m.load('data_dims.pik')
        else:
            if self._config.info:
                print("Checking a sample of dataset images for different dimensions...")

            for seg in random.sample(self.X,int(0.01*samples)):
                dims.add((samples,) + seg.getImgDim())
            cache_m.dump(dims,'data_dims.pik')

        l = list(dims)
        l.sort()
        return l

    def _release_data(self):
        del self. X
        del self.Y
        
        self.X = None
        self.Y = None
    
