#!/usr/bin/env python3
#-*- coding: utf-8

from Datasources.CellRep import CellRep

class RepCae(object):
    """
    This is the CAE model as implemented in:
    'Spatial Organization And Molecular Correlation Of Tumor-Infiltrating Lymphocytes Using Deep Learning On Pathology Images'
    Published in Cell Reports
    """

    def __init__(self):
        pass

    def build(self):
        """
        Builds and returns a trainable model. 
        """

        #Should we use serial or parallel model?
        if not parallel_model is None:
            model = parallel_model
        else:
            model = serial_model

