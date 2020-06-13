#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os

#Filter warnings
import warnings
warnings.filterwarnings('ignore')
    
import numpy as np

#Preparing migration to TF 2.0
import tensorflow as tf
if tf.__version__ >= '1.14.0':
    tf = tf.compat.v1
    from tensorflow.python.util import deprecation
    deprecation._PRINT_DEPRECATION_WARNINGS = False
    tf.logging.set_verbosity(tf.logging.ERROR)
    #tf.disable_v2_behavior()
    
from keras.layers import Average,Concatenate
from keras import backend as K
from keras.models import Model

#Locals
from Utils import CacheManager
from Models.GenericModel import GenericModel

class GenericEnsemble(GenericModel):
    """
    Defines common model attributes and methods

    Some models may implement a feature extractor or an ensemble build. If they do, the following methods
    will be available:
    - build_extractor
    - build_ensemble
    """
    def __init__(self,config,ds,name=None):
        super().__init__(config,ds,name=name)

    def is_ensemble(self):
        return self._config.strategy == 'EnsembleTrainer'

    def get_npweights_cache(self,add_ext=False):
        """
        Returns path to model cache.

        @param add_ext <boolean>: add numpy file extension to file name.
        """
        if add_ext:
            return "{}.npy".format(self.cache_m.fileLocation(self._weightsCache).split('.')[0])
        else:
            return self.cache_m.fileLocation(self._weightsCache).split('.')[0]

    def get_npmgpu_weights_cache(self,add_ext=False):
        """
        Returns path to model cache

        @param add_ext <boolean>: add numpy file extension to file name.
        """
        if add_ext:
            return "{}.npy".format(self.cache_m.fileLocation(self._mgpu_weightsCache).split('.')[0])
        else:
            return self.cache_m.fileLocation(self._mgpu_weightsCache).split('.')[0]
    
    def register_ensemble(self,m):
        self._model_n = m
        self._weightsCache = "{0}-EM{1}-weights.h5".format(self.name,m)
        self._mgpu_weightsCache = "{0}-EM{1}-mgpu-weights.h5".format(self.name,m)
        self._modelCache = "{0}-EM{1}-model.h5".format(self.name,m)
        
        self.cache_m.registerFile(os.path.join(self._config.weights_path,self._weightsCache),self._weightsCache)
        self.cache_m.registerFile(os.path.join(self._config.weights_path,self._mgpu_weightsCache),self._mgpu_weightsCache)
        self.cache_m.registerFile(os.path.join(self._config.model_path,self._modelCache),self._modelCache)


    def return_model_n(self):
        if hasattr(self,'_model_n'):
            return self._model_n
        else:
            return -1

    def build_extractor(self,**kwargs):
        """
        Builds a feature extractor.
        
        Weights should be loaded by caller!

        Key word arguments:
        preload_w: return model with weights already loaded? True -> Yes
        parallel: return parallel model (overrides gpu_count avaliation)? True -> Yes
        """
        #Weight loading for the feature extraction is done latter by requesting party
        kwargs['preload_w'] = False

        if self.is_ensemble():
            kwargs['npfile'] = True
            kwargs['feature'] = True
            s,p = self.build_ensemble(**kwargs)
            if 'parallel' in kwargs and not kwargs['parallel']:
                return (s,None)
            else:
                return (s,p)
            
        if 'parallel' in kwargs and not kwargs['parallel']:
            s,p = self._build(**kwargs)
            return (s,None)
        else:
            return self._build(**kwargs)        
        
    def build_ensemble(self,**kwargs):
        """
        Builds an ensemble of M Inception models.

        Weights are loaded here because of the way ensembles should be built.

        Default build: avareges the output of the corresponding softmaxes

        @param npfile <boolean>: loads weights from numpy files
        @param new <boolean>: build a new ensemble body or use the last built
        """

        if 'data_size' in kwargs:
            self.data_size = kwargs['data_size']
            
        if 'feature' in kwargs:
            feature = kwargs['feature']
        else:
            feature = False

        if 'npfile' in kwargs:
            npfile = kwargs['npfile']
        else:
            npfile = False            

        if 'new' in kwargs:
            new = kwargs['new']
        else:
            new = True
            
        if 'allocated_gpus' in kwargs and not kwargs['allocated_gpus'] is None:
            allocated_gpus = kwargs['allocated_gpus']
        else:
            allocated_gpus = self._config.gpu_count            

        inputs = None
        s_models = None
        p_models = None
        
        if new or not (hasattr(self,'_s_models') or hasattr(self,'_p_models')):
            if self._config.info and not new:
                print("[{}] No previous ensemble models stored, building new ones".format(self.name))
            s_models,p_models,inputs = self._build_ensemble_body(feature,npfile,allocated_gpus)
            self._s_models = s_models
            self._p_models = p_models
            self._en_inputs = inputs
        else:
            s_models = self._s_models
            p_models = self._p_models
            inputs = self._en_inputs

        s_outputs = [out for s in s_models for out in s.outputs]
        p_models = list(filter(lambda x: not x is None,p_models))
        if len(p_models) > 0:
            p_outputs = [out for p in p_models for out in p.outputs]
        else:
            p_outputs = None

        #Build the ensemble output from individual models
        s_model,p_model = None,None
        ##Single GPU model
        ## TODO: to enable full model reuse, we should convert feature extractor and classificator
        ## between one another
        if feature:
            x = Concatenate()(s_outputs)
        else:
            x = Average()(s_outputs)
        s_model = Model(inputs = inputs, outputs=x)

        ##Parallel model
        if not p_outputs is None:
            if feature:
                x = Concatenate()(p_outputs)
            else:
                x = Average()(p_outputs)
            p_model = Model(inputs=inputs,outputs=x)

        return s_model,p_model


    def _build_ensemble_body(self,feature,npfile,allocated_gpus):
        s_models = []
        p_models = []
        inputs = []
        
        width,height,channels = self._check_input_shape()
        if K.image_data_format() == 'channels_first':
            input_shape = (channels, height, width)
        else:
            input_shape = (height, width, channels)

        for m in range(self._config.emodels):
            self.register_ensemble(m)
            model,inp = self._build_architecture(input_shape=input_shape,training=False,
                                                      feature=feature,preload=False,
                                                      ensemble=True)

            inputs.append(inp)
            
            #Updates all layer names to avoid repeated name error
            for layer in model.layers:
                layer.name = 'EM{}-{}'.format(m,layer.name)
                
            single,parallel = self._configure_compile(model,allocated_gpus)
            
            single,parallel = self._load_weights(single,parallel,npfile,m)
            
            s_models.append(single)
            p_models.append(parallel)        

        return s_models,p_models,inputs    


    def _load_weights(self,single,parallel,npfile,m=''):
        if not parallel is None:
            #Updates all layer names to avoid repeated name error
            for layer in parallel.layers:
                layer.name = 'EM{}-{}'.format(m,layer.name)
            if npfile:
                parallel.set_weights(np.load(self.get_npmgpu_weights_cache(add_ext=True),allow_pickle=True))
                if self._config.info:
                    print("[{}] loaded ensemble weights: {}".format(self.name,self.get_npmgpu_weights_cache(add_ext=True)))
            elif os.path.isfile(self.get_mgpu_weights_cache()):                    
                parallel.load_weights(self.get_mgpu_weights_cache(),by_name=True)
                if self._config.info:
                    print("[{}] loaded ensemble weights: {}".format(self.name,self.get_mgpu_weights_cache()))
                    
            return single,parallel


        if npfile:
            single.set_weights(np.load(self.get_npweights_cache(add_ext=True),allow_pickle=True))
            if self._config.info:
                print("[{}] loaded ensemble weights: {}".format(self.name,self.get_npweights_cache(add_ext=True)))
        elif os.path.isfile(self.get_weights_cache()):
            single.load_weights(self.get_weights_cache(),by_name=True)
        else:
            if self._config.info:
                print("[{}] Could not load ensemble weights (model {})".format(self.name,m))
            single = None

        return single,parallel    
