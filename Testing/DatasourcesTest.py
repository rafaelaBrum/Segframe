#!/usr/bin/env python3
#-*- coding: utf-8

import cv2
import random

from Datasources.CellRep import CellRep

def run():
    #Run all tests below
    cr = CellRep('/Volumes/Trabalho/Doutorado/Dataset/Lymphocyte/TIL/test_patches/',verbose=1,pbar=True)
    X,Y = cr.load_metadata()

    #Check image dimensions
    print("Dataset has size: {0}".format(cr.get_dataset_dimensions()))
