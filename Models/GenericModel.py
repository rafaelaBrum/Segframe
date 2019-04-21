#!/usr/bin/env python3
#-*- coding: utf-8

from abc import ABC,abstractmethod

class GenericModel(ABC):
    """
    Defines common model attributes and methods
    """
    def __init__(self,config):
        self._config = config

    @abstractmethod
    def build(self):
        pass
