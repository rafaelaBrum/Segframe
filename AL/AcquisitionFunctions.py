#!/usr/bin/env python3
#-*- coding: utf-8

import numpy as np
import os
from tqdm import tqdm

from scipy.stats import mode

__doc__ = """
All acquisition functions should receive:
1 - numpy array of items
2 - numpy array of labels
3 - number of items to query
4 - keyword arguments specific for each function (if needed)

Returns: numpy array of element indexes
"""


def bayesian_varratios(pred_model,generator,data_size,**kwargs):
    """
    Calculation as defined in paper:
    Bayesian convolutional neural networks with Bernoulli approximate variational inference

    Function needs to extract the following configuration parameters:
    model <keras.Model>: model to use for predictions
    generator <keras.Sequence>: data generator for predictions
    data_size <int>: number of data samples
    mc_dp <int>: number of dropout iterations
    cpu_count <int>: number of cpu cores (used to define number of generator workers)
    gpu_count <int>: number of gpus available
    verbose <int>: verbosity level
    pbar <boolean>: user progress bars
    """

    if 'config' in kwargs:
        mc_dp = kwargs['config'].dropout_steps
        gpu_count = kwargs['config'].gpu_count
        cpu_count = kwargs['config'].cpu_count
        verbose = kwargs['config'].verbose
        pbar = kwargs['config'].progressbar
        query = kwargs['config'].acquire
    else:
        return None        

    All_Dropout_Classes = np.zeros(shape=(data_size,1))

    if pbar:
        l = tqdm(range(mc_dp), desc="MC Dropout",position=0)
    else:
        if kwargs['config'].info:
            print("Starting MC dropout sampling...")
        l = range(mc_dp)
                
    for d in l:
        if pbar:
            print("\n")
        elif kwargs['config'].info:
            print("Step {0}/{1}".format(d,mc_dp))
            
        proba = pred_model.predict_generator(generator,
                                                workers=3*cpu_count,
                                                max_queue_size=25*gpu_count,
                                                verbose=verbose)

        dropout_classes = proba.argmax(axis=-1)    
        dropout_classes = np.array([dropout_classes]).T
        All_Dropout_Classes = np.append(All_Dropout_Classes, dropout_classes, axis=1)

    Variation = np.zeros(shape=(data_size))

    for t in range(data_size):
        L = np.array([0])
        for d_iter in range(mc_dp):
            L = np.append(L, All_Dropout_Classes[t, d_iter+1])
        Predicted_Class, Mode = mode(L[1:])
        v = np.array(  [1 - Mode/float(mc_dp)])
        Variation[t] = v

    a_1d = Variation.flatten()
    x_pool_index = a_1d.argsort()[-query:][::-1]

    return x_pool_index

def bayesian_bald(pool_x,pool_y,query,kwargs):
    """
    Calculation as defined in paper:
    Bayesian convolutional neural networks with Bernoulli approximate variational inference
    """
    if 'model' in kwargs:
        model = kwargs['model']
    else:
        return None

    if 'config' in kwargs:
        mc_dp = kwargs['config'].dropout_steps
    else:
        mc_dp = 30

    batch_size = kwargs['config'].batch_size
    
