#!/usr/bin/env python3
#-*- coding: utf-8

import numpy as np
import os
import random

#Keras MNIST
from keras.datasets import mnist

#Local modules
from Datasources import GenericDatasource as gd
from Preprocessing import NPImage
from Utils import CacheManager

class MNIST(gd.GenericDS):
    """
    Class that parses label.txt text files and loads all images into memory
    """

    def __init__(self,data_path,keepImg=False,config=None):
        """
        @param data_path <str>: path to directory where image patches are stored
        @param config <argparse>: configuration object
        @param keepImg <boolean>: keep image data in memory
        """
        if data_path == '':
            data_path = os.path.join(os.path.expanduser('~'), '.keras','datasets')
            
        super().__init__(data_path,keepImg,config,name='MNIST')
        self.nclasses = 10


    def _load_metadata_from_dir(self,d):
        """
        Create NPImages from KERAS MNIST
        """
        class_set = set()
        t_x,t_y = ([],[])

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        tr_size = x_train.shape[0]
        test_size = x_test.shape[0]

        f_path = os.path.join(self.path,'mnist.npz')
        for s in range(tr_size):
            t_x.append(NPImage(f_path,x_train[s],True,'x_train',s,self._verbose))
            t_y.append(y_train[s])
            class_set.add(y_train[s])

        for s in range(test_size):
            t_x.append(NPImage(f_path,x_test[s],True,'x_test',s,self._verbose))
            t_y.append(y_test[s])
            class_set.add(y_test[s])

        return t_x,t_y
            
