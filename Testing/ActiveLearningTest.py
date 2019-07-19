#!/usr/bin/env python3
#-*- coding: utf-8

import random
from Datasources.CellRep import CellRep
from Trainers import ActiveLearningTrainer

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
    
def run(config):
    #Run all tests below
    config.data = 'CellRep'
    config.balance = True
    if config.local_test:
        config.predst = '/Volumes/Trabalho/Doutorado/Dataset/Lymphocyte/TIL/test_patches/'
    else:
        config.predst = '../data/lym_cnn_training_data/'
    
    #Start training
    trainer = ActiveLearningTrainer(config)

    #Test set balancing
    X,Y = trainer._ds().load_metadata()
    bX,bY = trainer._balance_classes(X,Y)

     
    
    trainer.run()
