#!/usr/bin/env python3
#-*- coding: utf-8

import sys
import os
import numpy as np


from WSIParse import TCGAMerger,GenericData
from Utils import Exitcodes
from Utils import CacheManager

def preprocess_data(config,img_types):
    """
    Main function in preprocessing.

    Works through estipulated configuration
    """

    #Check SRC and DST directories
    if not os.path.exists(config.presrc):
        if config.verbose > 0:
            print("[Preprocess] No such directory: {0}".format(config.presrc))
        sys.exit(Exitcodes.PATH_ERROR)
    if not os.path.exists(config.predst):
        os.makedirs(config.predst)

    #If SRC dir has already been scanned, no need to redo:
    #TODO: TCGA data has specific structure, other unordered sets should be considered.
    # think of a possible unified model!!
    cache_m = CacheManager.CacheManager(verbose=config.verbose)
    datatree = None
    if config.tcga:
        datatree = cache_m.load('tcga.pik')
        if datatree is None:
            datatree = TCGAMerger.Merger(config.presrc,config.verbose)
            cache_m.dump(datatree,'tcga.pik')
    else:
        datatree = cache_m.load('datatree.pik')
        if datatree is None:
            datatree = GenericData.GenericData(config.presrc,img_types)
