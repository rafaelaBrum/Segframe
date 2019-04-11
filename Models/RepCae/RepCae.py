#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
from Datasources.CellRep import CellRep

#Network
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import ZeroPadding2D,Convolution2D, MaxPooling2D
from keras.layers import AveragePooling2D
from keras.models import Model
from keras.layers import Activation, LeakyReLU,BatchNormalization
from keras.layers import Input,InputLayer
from keras import backend, optimizers
from keras.utils import multi_gpu_model
from keras import regularizers
from keras_contrib.layers import GroupNormalization
import keras
import numpy as np
import tensorflow as tf

class RepCae(object):
    """
    This is the CAE model as implemented in:
    'Spatial Organization And Molecular Correlation Of Tumor-Infiltrating Lymphocytes Using Deep Learning On Pathology Images'
    Published in Cell Reports
    """

    def __init__(self,config):
        """
        Configuration defined by user
        """
        self._config = config
        self._ds = None

    def build(self):
        """
        Builds and returns a trainable model. 
        """
        
        #Image shape by OpenCV reports height x width
        if not self._ds is None:
            dims = self._ds.get_dataset_dimensions()
        else:
            dims = [(None,100,100,3)]

        #Dataset may have images of different sizes. What to do? Currently, chooses the biggest....
        _,width,height,channels = 0,0,0,0
        for d in dims:
            if width < d[1] or length < d[2]:
                _,width,height,channels = d
            
        if backend.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)
            
        #Should we use serial or parallel model?
        if not parallel_model is None:
            model = parallel_model
        else:
            model = serial_model

        #Builds CAE model
        model = Sequential()

        #Layer 1
        model.add(Convolution2D(100, (5, 5),strides=1,padding='same',
                    kernel_initializer='he_normal',dilation_rate=1,input_shape=input_shape))
        model.add(LeakyRelu(alpha=0.2))
        model.add(GroupNormalization(groups=4,axis=-1))
        #Layer 2
        model.add(Convolution2D(120, (5, 5),strides=1,padding='same',kernel_initializer='he_normal'))
        model.add(LeakyRelu(alpha=0.2))
        model.add(GroupNormalization(groups=4,axis=-1))
        model.add(AveragePooling2D(pool_size=(2, 2),strides=2))
        #Layer 3
        model.add(Convolution2D(240, (3, 3),strides=1,padding='same',kernel_initializer='he_normal'))
        model.add(LeakyRelu(alpha=0.2))        
        model.add(GroupNormalization(groups=4,axis=-1))
        #Layer 4
        model.add(Convolution2D(320, (3, 3),strides=1,padding='same',kernel_initializer='he_normal'))
        model.add(LeakyRelu(alpha=0.2))        
        model.add(GroupNormalization(groups=4,axis=-1))
        model.add(AveragePooling2D(pool_size=(2, 2),strides=2))
        #Layer 5
        model.add(Convolution2D(640, (3, 3),strides=1,padding='same',kernel_initializer='he_normal'))
        model.add(LeakyRelu(alpha=0.2))        
        model.add(GroupNormalization(groups=4,axis=-1))
        #Layer 6
        model.add(Convolution2D(1024, (3, 3),strides=1,padding='same',kernel_initializer='he_normal'))
        model.add(LeakyRelu(alpha=0.2))        
        model.add(GroupNormalization(groups=4,axis=-1))
        model.add(AveragePooling2D(pool_size=(2, 2),strides=2))

        #Linearize - END Encoder
        model.add(Convolution2D(640, (1, 1),strides=1,padding='same',kernel_initializer='he_normal'))
        model.add(LeakyRelu(alpha=0.2))        
        model.add(GroupNormalization(groups=4,axis=-1))
        model.add(Convolution2D(100, (1, 1),strides=1,padding='same',kernel_initializer='he_normal',name='feat_map'))
        model.add(Activation('relu'))        
        model.add(GroupNormalization(groups=4,axis=-1))
        model.add(Convolution2D(100, (1, 1),strides=1,padding='same',kernel_initializer='he_normal'))
        model.add(LeakyRelu(alpha=0.2))        
        model.add(GroupNormalization(groups=4,axis=-1))
        model.add(Convolution2D(1, (1, 1),strides=1,padding='same',kernel_initializer='he_normal'))
        
    def load_data(self):
        """
        Model knows which data it should load
        """
        if self._config.data:
            self._ds = importlib.import_module('Datasources',self._config.data)()
        else:
            self._ds = CellRep()

        t_x,t_y = self._ds.load_data(self._config.split)
        return t_x,t_y
        
