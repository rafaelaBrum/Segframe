#!/usr/bin/env python3
#-*- coding: utf-8

__all__ = ['VGG16','UNet','Inception','SmallNet']

from .VGG import VGG16,VGG16A2,VGG16A3
from .VGG import BayesVGG16, BayesVGG16A2
from .KMNIST import KNet,BayesKNet,GalKNet
from .EKNet import BayesEKNet
from .InceptionV4 import Inception,EFInception
from .ALTransf import SmallNet
