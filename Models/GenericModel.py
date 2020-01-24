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

    def is_ensemble(self):
        return self._config.strategy == 'EnsembleTrainer'

    def get_ds(self):
        return self._ds
    
    @abstractmethod
    def build(self,**kwargs):
        """
        Possible parameters to pass in kwargs (some models may use them, others not):
        data_size <int>: Training data size, usefull for custom parameter settings
        pre_trained <boolean>: Should load a pre-trained model?
        """
        pass

    @abstractmethod
    def get_model_cache(self):
        pass
    
    @abstractmethod
    def get_weights_cache(self):
        pass
    
    @abstractmethod
    def get_mgpu_weights_cache(self):
        pass
