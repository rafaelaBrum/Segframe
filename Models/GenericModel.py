#!/usr/bin/env python3
#-*- coding: utf-8

from abc import ABC,abstractmethod

class GenericModel(ABC):
    """
    Defines common model attributes and methods

    Some models may implement a feature extractor or an ensemble build. If they do, the following methods
    will be available:
    - build_extractor
    - build_ensemble
    """
    def __init__(self,config,ds,name):
        self._config = config
        self._ds = ds
        self.name = name
        self.single = None
        self.parallel = None

    def _check_input_shape(self):
        #Image shape by OpenCV reports height x width
        if not self._config.tdim is None:
            if len(self._config.tdim) == 2:
                dims = [(None,) + tuple(self._config.tdim) + (3,)]
            else:
                dims = [(None,) + tuple(self._config.tdim)]
        elif not self._ds is None:
            dims = self._ds.get_dataset_dimensions()
        else:
            dims = [(None,100,100,3)]

        #Dataset may have images of different sizes. What to do? Currently, chooses the smallest....
        _,width,height,channels = dims[0]

        return (width,height,channels)

    @abstractmethod
    def get_model_cache(self):
        pass
    
    @abstractmethod
    def get_weights_cache(self):
        pass
    
    @abstractmethod
    def get_mgpu_weights_cache(self):
        pass
    
    @abstractmethod
    def _build(self,**kwargs):
        pass
        
    def build(self,**kwargs):
        """
        Child classes should implement: _build method

        Optional params:
        @param data_size <int>: size of the training dataset
        @param training <boolean>: set layer behavior to training mode (aplicable to dropout/BatchNormalization)
        @param feature <boolean>: return features instead of softmax classification
        @param preload_w <boolean>: load pre-trained weights to model
        @param allocated_gpus <int>: number of GPU availables
        @param pre_trained <boolean>: returned model should be pre-trained or not
        """

        width,height,channels = self._check_input_shape()

        if 'data_size' in kwargs:
            self.data_size = kwargs['data_size']

        if 'training' in kwargs:
            training = kwargs['training']
        else:
            training = None
            
        if 'feature' in kwargs:
            feature = kwargs['feature']
        else:
            feature = False

        if 'preload_w' in kwargs:
            preload = kwargs['preload_w']
        else:
            preload = False

        if not 'allocated_gpus' in kwargs or kwargs['allocated_gpus'] is None:
            kwargs['allocated_gpus'] = self._config.gpu_count
            
        model,parallel_model = self._build(width,height,channels,**kwargs)
        
        self.single = model
        self.parallel = parallel_model
        
        return (model,parallel_model)

    def get_ds(self):
        return self._ds
        
