#!/usr/bin/env python3
#-*- coding: utf-8

import random
import os
import numpy as np
from matplotlib import pyplot as plt
from Datasources.CellRep import CellRep

def run(config):
    #Run all tests below
    config.data = 'CellRep'
    if config.local_test:
        config.predst = '/Volumes/Trabalho/Doutorado/Dataset/Lymphocyte/TIL/test_patches/'
    else:
        config.predst = '../data/lym_cnn_training_data/'

    print("Verbosity: {0}".format(config.verbose))

    cr = CellRep(config.predst,keepImg=True,config=config)
    X,Y = cr.load_metadata()
    #Check image dimensions
    print("Dataset has size(s): {0}".format(cr.get_dataset_dimensions()))

    run_all(config,cr)
    
    if config.local_test:
        run_local(config,cr,X,Y)
        
def run_all(config,cr):
    #Split dataset
    dataset = cr.split_metadata(split=config.split)
    print("Train set: {0} items, {1} labels".format(len(dataset[0][0]),len(dataset[0][1])))
    unique,count = np.unique(dataset[0][1],return_counts=True)
    l_count = dict(zip(unique,count))
    print("Train labels: {0} are 0; {1} are 1;".format(l_count[0],l_count[1]))

    print("Validate set: {0} items, {1} labels".format(len(dataset[1][0]),len(dataset[1][1])))
    unique,count = np.unique(dataset[1][1],return_counts=True)
    l_count = dict(zip(unique,count))
    print("Validade labels: {0} are 0; {1} are 1;".format(l_count[0],l_count[1]))
    
    print("Test set: {0} items, {1} labels".format(len(dataset[2][0]),len(dataset[2][1])))
    unique,count = np.unique(dataset[2][1],return_counts=True)
    l_count = dict(zip(unique,count))
    print("Test labels: {0} are 0; {1} are 1;".format(l_count[0],l_count[1]))    

    check_labels(config,cr)
    check_data_split(dataset)

def run_local(config,cr,X,Y):
    """
    Full dataset is too big for these
    """
    d_x,d_y = cr.load_data()
    print("X size: {0} \n ************* \n".format(len(X)))
    print("Y size: {0} \n ************* \n".format(len(Y)))

    #Show a random patch
    img = d_x[random.randint(0,len(d_x) - 1)]
    print("Image array shape: {0}; dtype: {1}".format(d_x.shape,d_x.dtype))
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    
def check_data_split(split_data):
    """
    Checks if there is no reocurrence of samples between sets
    """
    train,val,test = split_data
    x_train,y_train = train
    x_val,y_val = val
    x_test,y_test = test
    
    val_d = {x_val[i]:y_val[i] for i in range(len(x_val))}
    test_d = {x_test[i]:y_test[i] for i in range(len(x_test))}
    for s in x_train:
        if s in val_d:
            print("Item {0} of training set is also in validation set".format(s))
        if s in test_d:
            print("Item {0} of training set is also in test set".format(s))
    print("Done checking data split")
        
def check_labels(config,ds):
    """
    Checks all labels from a sequential run against the ones produced by the Datasource classes.
    """
    files = os.listdir(config.predst)
    dlist = []
    
    for f in files:
        item = os.path.join(config.predst,f)
        if os.path.isdir(item):
            dlist.append(item)
    reference = {}
    count = 0
    item_c = len(dlist)
    for item in dlist:
        t_x,t_y = ds._load_metadata_from_dir(item)
        t_dct = {t_x[i]:t_y[i] for i in range(len(t_x))}
        reference.update(t_dct)
        if config.info:
            print("Processing dirs sequentialy ({0}/{1})".format(count,item_c))
        count += 1
            
    #Now the DS metadata
    X2,Y2 = ds.load_metadata()

    for j in range(len(X2)):
        if reference[X2[j]] != Y2[j]:
            print("Item labels differ.\n - Reference item: {0};\n - Reference label: {1};\n Metadata label: {2}".format(
                X2[j],reference[X2[j]],y2[j]))

    print("If no messages reporting misslabeling was displayed, everything is good.")
