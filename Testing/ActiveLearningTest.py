#!/usr/bin/env python3
#-*- coding: utf-8

import random
from Datasources.CellRep import CellRep
from Trainers import ALTrainer

def run(config):
    #Run all tests below
    config.data = 'CellRep'
    if config.local_test:
        config.predst = '/Volumes/Trabalho/Doutorado/Dataset/Lymphocyte/TIL/test_patches/'
    else:
        config.predst = '../data/lym_cnn_training_data/'
    
    #Start training
    trainer = ALTrainer(config)
    trainer.run()
