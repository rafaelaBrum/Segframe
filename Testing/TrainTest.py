#!/usr/bin/env python3
#-*- coding: utf-8

from Datasources import CellRep

def run():
    #Run all tests below
    cr = CellRep('/Volumes/Trabalho/Doutorado/Dataset/Lymphocyte/TIL/test_patches/luad_validation_1',1,True)
    X,Y = cr.load_metadata()
    print("X: {0} \n ************* \n".format(X))
    print("Y: {0} \n ************* \n".format(Y))
