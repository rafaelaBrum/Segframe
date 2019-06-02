#!/usr/bin/env python3
#-*- coding: utf-8

import random
from Datasources.CellRep import CellRep
from Models import Predictor
from Models.Predictions import print_prediction

def run(config):
    #Run all tests below
    config.data = 'CellRep'
    if config.local_test:
        config.predst = '/Volumes/Trabalho/Doutorado/Dataset/Lymphocyte/TIL/test_patches/'
    else:
        config.predst = '../data/lym_cnn_training_data/'
        
    #Start predictions
    pred = Predictor(config)
    pred.run()
    print_prediction(config)
