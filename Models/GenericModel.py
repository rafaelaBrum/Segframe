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
        if not self._ds is None:
            dims = self._ds.get_dataset_dimensions()
        else:
            dims = [(None,100,100,3)]

        #Dataset may have images of different sizes. What to do? Currently, chooses the biggest....
        _,width,height,channels = 0,0,0,0
        for d in dims:
            if width < d[1] or length < d[2]:
                _,width,height,channels = d

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
