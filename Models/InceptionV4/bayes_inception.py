#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os

import numpy as np
import tensorflow as tf

#Network
from keras.models import Sequential,Model
from keras.layers import Input
from keras import backend, optimizers
from keras.utils import multi_gpu_model
from keras import backend as K

#Locals
from Utils import CacheManager
from Models.GenericModel import GenericModel

class BayesInception(GenericModel):
    """
    Implements abstract methods from GenericModel.
    Model is the same as in: https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_resnet_v2.py
    Addapted to provide a Bayesian model
    """
    def __init__(self,config,ds,name=None):
        super().__init__(config,ds,name=name)
        if name is None:
            self.name = "BayesInception"
        self._modelCache = "{0}-model.h5".format(self.name)
        self._weightsCache = "{0}-weights.h5".format(self.name)
        self._mgpu_weightsCache = "{0}-mgpu-weights.h5".format(self.name)
 
        self.cache_m = CacheManager()
        self.cache_m.registerFile(os.path.join(config.model_path,self._modelCache),self._modelCache)
        self.cache_m.registerFile(os.path.join(config.weights_path,self._weightsCache),self._weightsCache)
        self.cache_m.registerFile(os.path.join(config.weights_path,self._mgpu_weightsCache),self._mgpu_weightsCache)

        self.single = None
        self.parallel = None
        
    def get_model_cache(self):
        """
        Returns path to model cache
        """
        return self.cache_m.fileLocation(self._modelCache)
    
    def get_weights_cache(self):
        """
        Returns path to model cache
        """
        return self.cache_m.fileLocation(self._weightsCache)

    def get_mgpu_weights_cache(self):
        """
        Returns path to model cache
        """
        return self.cache_m.fileLocation(self._mgpu_weightsCache)
        
    def build(self,**kwargs):

        model,parallel_model = self._build(**kwargs)
        
        self.single = model
        self.parallel = parallel_model
        
        return (model,parallel_model)
    
    def _build(self,**kwargs):
        """
        @param pre_trained <boolean>: returned model should be pre-trained or not
        @param data_size <int>: size of the training dataset
        """
        width,height,channels = self._check_input_shape()

        if 'data_size' in kwargs:
            self.data_size = kwargs['data_size']

        if 'training' in kwargs:
            training = kwargs['training']
        else:
            training = True
            
        if 'feature' in kwargs:
            feature = kwargs['feature']
        else:
            feature = False

        if 'pre_load_w' in kwargs:
            preload = kwargs['pre_load_w']
        else:
            preload = True
            
        if backend.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)

        self.cache_m = CacheManager()
        
        model = self._build_architecture(input_shape,training,feature,preload)
        
        #Check if previous training and LR is saved, if so, use it
        lr_cache = "{0}_learning_rate.txt".format(self.name)
        self.cache_m.registerFile(os.path.join(self._config.cache,lr_cache),lr_cache)
        l_rate = 0.0005
        if os.path.isfile(self.cache_m.fileLocation(lr_cache)) and not self._config.new_net:
            l_rate = float(self.cache_m.read(lr_cache))
            if self._config.info:
                print("Found previous learning rate: {0}".format(l_rate))
        
        #opt = optimizers.SGD(lr=l_rate, decay=1.5e-4, momentum=0.9, nesterov=True)
        opt = optimizers.Adam(lr = l_rate)
        #opt = optimizers.Adadelta(lr=l_rate)

        #Return parallel model if multiple GPUs are available
        parallel_model = None
        
        if self._config.gpu_count > 1:
            with tf.device('/cpu:0'):
                model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

            parallel_model = multi_gpu_model(model,gpus=self._config.gpu_count)
            parallel_model.compile(loss='categorical_crossentropy',
                                       optimizer=opt,
                                       metrics=['accuracy'],
                                       #options=p_opt, 
                                       #run_metadata=p_mtd
                                       )
        else:
            model.compile(loss='categorical_crossentropy',
                optimizer=opt,
                metrics=['accuracy'],
                #options=p_opt, 
                #run_metadata=p_mtd
                )

        return (model,parallel_model)

    def _build_architecture(self,input_shape,training=None,feature=False,preload=True):
        from . import inception_resnet_v2

        kwargs = {'training':training,
                    'feature':feature,
                    'custom_top':True,
                    'preload':preload,
                    'batch_n':True if self._config.gpu_count <= 1 else False}
        
        inp = Input(shape=input_shape)
                
        inception_body = inception_resnet_v2.InceptionResNetV2(include_top=False,
                                                                weights='imagenet',
                                                                input_tensor=inp,
                                                                input_shape=input_shape,
                                                                pooling='avg',
                                                                classes=self._ds.nclasses,
                                                                **kwargs)
        

        return inception_body
                                
