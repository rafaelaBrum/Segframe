#!/usr/bin/env python3
#-*- coding: utf-8

import numpy as np
import os

#Local modules
from Datasources import GenericDatasource as gd
from Preprocessing import GenericImage

class CellRep(gd.GenericDS):
    """
    Class that parses label.txt text files and loads all images into memory
    """

    def __init__(self,data_path,verbose=0,pbar=False):
        """
        @param data_path <str>: path to directory where image patches are stored
        @param verbose <int>: verbosity level
        @param pbar <boolean>: display progress bars
        """
        super().__init__(path,verbose,pbar)


    def _load_metadata_from_dir(self,d):
        """
        Create SegImages from a directory
        """
        
        labels = open(os.path.join(d,'label.txt'),'r')

        t_x,t_y = ([],[])
        for f in labels:
            tmp = f.strip().split()
            f_name,f_label = tmp[0],tmp[1]
            origin = tmp[2]
            coord = (tmp[3],tmp[4])
            seg = GenericImage(os.path.join(d,f_name),origin,coord,verbose=verbose)
            t_x.append(seg)
            t_y.append(int(f_label))

        return t_x,t_y

    def _release_data(self):
        del self. X
        del self.Y
        
        self.X = None
        self.Y = None
    
