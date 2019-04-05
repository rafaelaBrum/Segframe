#!/usr/bin/env python3
#-*- coding: utf-8

import cv2
import random
from matplotlib import pyplot as plt
from Datasources.CellRep import CellRep

def run():
    #Run all tests below
    cr = CellRep('/Volumes/Trabalho/Doutorado/Dataset/Lymphocyte/TIL/test_patches/',verbose=1,pbar=True)
    X,Y = cr.load_metadata()
    print("X size: {0} \n ************* \n".format(len(X)))
    print("Y size: {0} \n ************* \n".format(len(Y)))

    d_x,d_y = cr.load_data(keepImg=True)
    
    #Show a random patch
    img = d_x[random.randint(0,len(d_x) - 1)]
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

    #Split test
    dataset = cr.load_data(split=(0.8,0.1,0.1))
    print("Dataset size: {0}:\n - Training set size: {1};\n - Validation set size: {2};\n - Test set size:{3}".format(
        len(d_x),len(dataset[0][0]),len(dataset[1][0]),len(dataset[2][0])))
    
    #Start training
    
