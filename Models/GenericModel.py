#!/usr/bin/env python3
#-*- coding: utf-8

from abc import ABC,abstractmethod

class GenericModel(ABC):
    """
    Defines common model attributes and methods
    """
    def __init__(self,config,ds,name):
        self._config = config
        self._ds = ds
        self.name = name

    def _check_input_shape(self):
        #Image shape by OpenCV reports height x width
        if not self._config.tdim is None:
            dims = [(None,) + tuple(self._config.tdim) + (3,)]
        elif not self._ds is None:
            dims = self._ds.get_dataset_dimensions()
        else:
            dims = [(None,100,100,3)]

        #Dataset may have images of different sizes. What to do? Currently, chooses the smallest....
        _,width,height,channels = dims[0]

        return (width,height,channels)
            
    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def get_model_cache(self):
        pass
    
    @abstractmethod
    def get_weights_cache(self):
        pass

    def get_weights_cache_mgpu(self):
        return None
