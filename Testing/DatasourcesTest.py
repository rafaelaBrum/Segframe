#!/usr/bin/env python3
#-*- coding: utf-8

import cv2
import random

from Datasources.CellRep import CellRep

def run():
    #Run all tests below
    cr = CellRep('/Volumes/Trabalho/Doutorado/Dataset/Lymphocyte/TIL/test_patches/',keepImg=True,verbose=1,pbar=True)
    X,Y = cr.load_metadata()

    #Check image dimensions
    print("Dataset has size: {0}".format(cr.get_dataset_dimensions()))

    #Split dataset
    dataset = cr.load_data(split=(0.8,0.1,0.1))
    print("Train set: {0} items, {1} labels".format(len(dataset[0][0]),len(dataset[0][1])))
    print("Validate set: {0} items, {1} labels".format(len(dataset[1][0]),len(dataset[1][1])))
    print("Test set: {0} items, {1} labels".format(len(dataset[2][0]),len(dataset[2][1])))    
