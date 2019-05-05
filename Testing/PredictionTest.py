#!/usr/bin/env python3
#-*- coding: utf-8

import random
from Datasources.CellRep import CellRep
from Models import Predictor
from Models.Predictions import print_previous_prediction

def run(config):
    #Run all tests below
    config.data = 'CellRep'
    config.predst = '/Volumes/Trabalho/Doutorado/Dataset/Lymphocyte/TIL/test_patches/'
    config.network = 'VGG16'

    #Start predictions
    pred = Predictor(config)
    pred.run()
    print_previous_prediction(config)
