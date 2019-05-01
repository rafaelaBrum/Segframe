#!/usr/bin/env python3
#-*- coding: utf-8

import cv2
import random
from Datasources.CellRep import CellRep
from Models import Trainer

def run(config):
    #Run all tests below
    config.data = 'CellRep'
    config.predst = '/Volumes/Trabalho/Doutorado/Dataset/Lymphocyte/TIL/test_patches/'
    config.network = 'VGG16'
    
    #Start training
    trainer = Trainer(config)
    trainer.run()
    
