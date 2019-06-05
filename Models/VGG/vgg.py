#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os

import numpy as np
import tensorflow as tf

#Network
from keras.models import Sequential,Model
from keras.layers import Input,Activation
from keras.layers import Dense, Dropout, Flatten
from keras.layers import ZeroPadding2D,Convolution2D, MaxPooling2D
from keras import backend, optimizers
from keras.utils import multi_gpu_model
from keras.applications import vgg16
from keras import regularizers
from keras_contrib.layers import GroupNormalization
from keras import backend as K

#Locals
from Utils import CacheManager
from Models.GenericModel import GenericModel

class VGG16(GenericModel):
    """
    Implements abstract methods from GenericModel.
    Producess a VGG16 model as implemented by Keras, with convolutional layers
    FC layers are substituted by Conv2D, as defined in:
    https://github.com/ALSM-PhD/quip_classification/blob/master/NNFramework_TF/sa_networks/vgg.py
    """
    def __init__(self,config,ds,name=None):
        super().__init__(config,ds,name=name)
        if name is None:
            self.name = "VGG16_A1"
        self._modelCache = "{0}-model.h5".format(self.name)
        self._weightsCache = "{0}-weights.h5".format(self.name)
        
        self.cache_m = CacheManager()
        self.cache_m.registerFile(os.path.join(config.model_path,self._modelCache),self._modelCache)
        self.cache_m.registerFile(os.path.join(config.weights_path,self._weightsCache),self._weightsCache)

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
    
    def build(self,pre_trained=False):
        """
        Returns a VGG 16 model instance, final fully-connected layers are substituted by Conv2Ds
        
        @param pre_trained <boolean>: returned model should be pre-trained or not
        """
        width,height,channels = self._check_input_shape()
        
        if backend.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)

            
        self.cache_m = CacheManager()
        
        model = self._build_architecture(input_shape)
        
        #Check if previous training and LR is saved, if so, use it
        lr_cache = "{0}_learning_rate.txt".format(self.name)
        self.cache_m.registerFile(os.path.join(self._config.cache,lr_cache),lr_cache)
        l_rate = 0.00005
        if os.path.isfile(self.cache_m.fileLocation(lr_cache)) and not self._config.new_net:
            l_rate = float(self.cache_m.read(lr_cache))
            if self._config.info:
                print("Found previous learning rate: {0}".format(l_rate))
        
        sgd = optimizers.SGD(lr=l_rate, decay=1.5e-4, momentum=0.9, nesterov=True)
        #adam = optimizers.Adam(lr = l_rate)
        
        #Return parallel model if multiple GPUs are available
        parallel_model = None
        
        if self._config.gpu_count > 1:
            with tf.device('/cpu:0'):
                model.compile(loss='categorical_crossentropy',
                    optimizer=sgd,
                    metrics=['accuracy'])

            parallel_model = multi_gpu_model(model,gpus=self._config.gpu_count)
            parallel_model.compile(loss='categorical_crossentropy',
                                       optimizer=sgd,
                                       metrics=['accuracy'],
                                       #options=p_opt, 
                                       #run_metadata=p_mtd
                                       )
        else:
            model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'],
                #options=p_opt, 
                #run_metadata=p_mtd
                )

        return (model,parallel_model)

    def _build_architecture(self,input_shape):
        original_vgg16 = vgg16.VGG16(weights=self.cache_m.fileLocation('vgg16_weights_notop.h5'),
                                         include_top=False,
                                         input_shape=input_shape)

        #Freeze initial layers, except for the last 3:
        #for layer in original_vgg16.layers[:-2]:
        #    layer.trainable = False
            
        model = Sequential()
        model.add(original_vgg16)
        model.add(Convolution2D(4096, (7, 7),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.75))
        model.add(Convolution2D(4096, (1, 1),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.75))
        model.add(Convolution2D(self._ds.nclasses, (1, 1),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Flatten())
        model.add(Dense(self._ds.nclasses))
        model.add(Activation('softmax'))

        return model

class VGG16A2(VGG16):
    """
    VGG variation, uses GroupNormalization and more dropout
    """
    def __init__(self,config,ds):
        super(VGG16A2,self).__init__(config=config,ds=ds,name = "VGG16_A2")

    def _build_architecture(self,input_shape):

        original_vgg16 = vgg16.VGG16(weights=self.cache_m.fileLocation('vgg16_weights_notop.h5'),
                                         include_top=False,
                                         input_shape=input_shape)
        layer_dict = dict([(layer.name, layer) for layer in original_vgg16.layers])

        model = Sequential()
        x = Convolution2D(64, (3, 3),input_shape=input_shape,
                    strides=1,
                    padding='valid',
                    name='block1_conv1',
                    weights=layer_dict['block1_conv1'].get_weights(),
                    kernel_regularizer=regularizers.l2(0.0005))
        model.add(x)
        model.add(GroupNormalization(groups=4,axis=-1))
        model.add(Activation('relu'))
        #model.add(Dropout(0.1))
    
        #Second layer
        model.add(ZeroPadding2D(padding=1))
        x = Convolution2D(64, (3, 3),strides=1,
                    padding='valid',
                    name='block1_conv2',
                    weights=layer_dict['block1_conv2'].get_weights(),
                    kernel_regularizer=regularizers.l2(0.0005))
        model.add(x)
        model.add(GroupNormalization(groups=4,axis=-1))        
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        #model.add(Dropout(0.1))
        
        #Third layer
        model.add(ZeroPadding2D(padding=1))
        x = Convolution2D(128, (3, 3),strides=1,
                    padding='valid',
                    name='block2_conv1',
                    weights=layer_dict['block2_conv1'].get_weights(),
                    kernel_regularizer=regularizers.l2(0.0005))
        model.add(x)
        model.add(GroupNormalization(groups=4,axis=-1))        
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))
        
        #Fourth layer
        model.add(ZeroPadding2D(padding=1))
        x = Convolution2D(128, (3, 3),strides=1,
                padding='valid',
                name='block2_conv2',
                weights=layer_dict['block2_conv2'].get_weights(),
                kernel_regularizer=regularizers.l2(0.0005))
        model.add(x)
        model.add(GroupNormalization(groups=4,axis=-1))        
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        #model.add(Dropout(0.2))
        
        #Fifth layer
        model.add(ZeroPadding2D(padding=1))
        x = Convolution2D(256, (3, 3),strides=1,
                padding='valid',
                name='block3_conv1',
                weights=layer_dict['block3_conv1'].get_weights(),
                kernel_regularizer=regularizers.l2(0.0005))
        model.add(x)
        model.add(GroupNormalization(groups=4,axis=-1))        
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))
        
        #Sith layer
        model.add(ZeroPadding2D(padding=1))
        x = Convolution2D(256, (3, 3),strides=1,
                padding='valid',
                name='block3_conv2',
                weights=layer_dict['block3_conv2'].get_weights(),
                kernel_regularizer=regularizers.l2(0.0005))
        model.add(x)
        model.add(GroupNormalization(groups=4,axis=-1))        
        model.add(Activation('relu'))
        #model.add(Dropout(0.3))
        
        #Seventh layer
        model.add(ZeroPadding2D(padding=1))
        x = Convolution2D(256, (3, 3),strides=1,
            padding='valid',
            name='block3_conv3',
            weights=layer_dict['block3_conv3'].get_weights(),
            kernel_regularizer=regularizers.l2(0.0005))
        model.add(x)
        model.add(GroupNormalization(groups=4,axis=-1))        
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        #model.add(Dropout(0.3))
        
        #Eigth layer
        model.add(ZeroPadding2D(padding=1))
        x = Convolution2D(512, (3, 3),strides=1,
            padding='valid',
            name='block4_conv1',
            weights=layer_dict['block4_conv1'].get_weights(),
            kernel_regularizer=regularizers.l2(0.0005))
        model.add(x)
        model.add(GroupNormalization(groups=4,axis=-1))        
        model.add(Activation('relu'))
        #model.add(Dropout(0.3))
        
        #Nineth layer
        model.add(ZeroPadding2D(padding=1))
        x = Convolution2D(512, (3, 3),strides=1,
            padding='valid',
            name='block4_conv2',
            weights=layer_dict['block4_conv2'].get_weights(),
            kernel_regularizer=regularizers.l2(0.0005))
        model.add(x)
        model.add(GroupNormalization(groups=4,axis=-1))        
        model.add(Activation('relu'))
        #model.add(Dropout(0.3))
        
        #Tenth layer
        model.add(ZeroPadding2D(padding=1))
        x = Convolution2D(512, (3, 3),strides=1,
            padding='valid',
            name='block4_conv3',
            weights=layer_dict['block4_conv3'].get_weights(),
            kernel_regularizer=regularizers.l2(0.0005))
        model.add(x)
        model.add(GroupNormalization(groups=4,axis=-1))        
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        #model.add(Dropout(0.3))
        
        #Eleventh layer
        model.add(ZeroPadding2D(padding=1))
        x = Convolution2D(512, (3, 3),strides=1,
            padding='valid',
            name='block5_conv1',
            #kernel_initializer='he_normal',
            weights=layer_dict['block5_conv1'].get_weights(),
            kernel_regularizer=regularizers.l2(0.0005))
        model.add(x)
        model.add(GroupNormalization(groups=4,axis=-1))        
        model.add(Activation('relu'))
        #model.add(Dropout(0.3))
        
        #Twelth layer
        model.add(ZeroPadding2D(padding=1))
        x = Convolution2D(512, (3, 3),strides=1,
            padding='valid',
            name='block5_conv2',
            #kernel_initializer='he_normal',
            weights=layer_dict['block5_conv2'].get_weights(),
            kernel_regularizer=regularizers.l2(0.0005))
        model.add(x)
        model.add(GroupNormalization(groups=4,axis=-1))
        model.add(Activation('relu'))
        #model.add(Dropout(0.3))
        
        #Thirtenth layer
        model.add(ZeroPadding2D(padding=1))
        x = Convolution2D(512, (3, 3),strides=1,
            padding='valid',
            name='block5_conv3',
            #kernel_initializer='he_normal',
            weights=layer_dict['block5_conv3'].get_weights(),
            kernel_regularizer=regularizers.l2(0.0005))
        model.add(x)
        model.add(GroupNormalization(groups=4,axis=-1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        #model.add(Dropout(0.3))

        #Freeze initial layers, except for the last 3:
        #for layer in model.layers[:-3]:
        #    layer.trainable = False

        #Last FC layers are substituted by Conv layers
        model.add(Convolution2D(4096, (7, 7),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.75))
        model.add(Convolution2D(4096, (1, 1),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        #model.add(Convolution2D(2, (1, 1)))
        model.add(Flatten())
        model.add(Dropout(0.75))
        model.add(Dense(self._ds.nclasses))
        model.add(Activation('sigmoid'))
        
        return model

class VGG16A3(VGG16):
    """
    VGG variation, uses GroupNormalization and more dropout
    """
    def __init__(self,config,ds):
        super(VGG16A3,self).__init__(config=config,ds=ds,name = "VGG16_A3")

    def _build_architecture(self,input_shape):
        original_vgg16 = vgg16.VGG16(weights=self.cache_m.fileLocation('vgg16_weights_notop.h5'),
                                         include_top=False,
                                         input_shape=input_shape)

        #Freeze initial layers, except for the last 3:
        for layer in original_vgg16.layers[:-6]:
            layer.trainable = False
            
        model = Sequential()
        model.add(original_vgg16)
        model.add(Dropout(0.4))
        model.add(Convolution2D(4096, (5, 5),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.4))
        model.add(Convolution2D(4096, (1, 1),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Activation('relu'))
        #model.add(Convolution2D(2, (1, 1)))
        model.add(Dropout(0.4))
        model.add(Convolution2D(self._ds.nclasses, (1, 1),strides=1,padding='valid',kernel_initializer='he_normal'))
        model.add(Flatten())
        model.add(Dense(self._ds.nclasses))
        model.add(Activation('softmax'))
        
        return model
