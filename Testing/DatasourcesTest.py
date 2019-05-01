#!/usr/bin/env python3
#-*- coding: utf-8

import cv2
import random
from matplotlib import pyplot as plt

from Datasources.CellRep import CellRep

def run(config):
    #Run all tests below
    config.data = 'CellRep'
    config.predst = '/Volumes/Trabalho/Doutorado/Dataset/Lymphocyte/TIL/test_patches/'

    cr = CellRep(config.predst,keepImg=True,config=config)
    X,Y = cr.load_metadata()
    d_x,d_y = cr.load_data()
    print("X size: {0} \n ************* \n".format(len(X)))
    print("Y size: {0} \n ************* \n".format(len(Y)))

    #Check image dimensions
    print("Dataset has size: {0}".format(cr.get_dataset_dimensions()))

     #Show a random patch
    img = d_x[random.randint(0,len(d_x) - 1)]
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
     
    #Split dataset
    dataset = cr.load_data(split=(0.8,0.1,0.1))
    print("Train set: {0} items, {1} labels".format(len(dataset[0][0]),len(dataset[0][1])))
    print("Validate set: {0} items, {1} labels".format(len(dataset[1][0]),len(dataset[1][1])))
    print("Test set: {0} items, {1} labels".format(len(dataset[2][0]),len(dataset[2][1])))    
