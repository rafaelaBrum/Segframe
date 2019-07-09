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


def bayesian_varratios(data,query,kwargs):
    """
    Calculation as defined in paper:
    Bayesian convolutional neural networks with Bernoulli approximate variational inference

    @param data <tuple>: X,Y as numpy arrays
    """
    from keras.preprocessing.image import ImageDataGenerator
    from Trainers import ThreadedGenerator
    
    if 'model' in kwargs:
        model = kwargs['model']
    else:
        return None

    if 'config' in kwargs:
        mc_dp = kwargs['config'].dropout_steps
        gpu_count = kwargs['config'].gpu_count
        batch_norm = kwargs['config'].batch_norm
        batch_size = kwargs['config'].batch_size
        fix_dim = kwargs['config'].tdim
        verbose = kwargs['config'].verbose
        pbar = kwargs['config'].progressbar
    else:
        return None

    if 'ds' in kwargs:
        ds = kwargs['ds']
    else:
        return None
        
    if fix_dim is None:
        fix_dim = ds.get_dataset_dimensions()[0][1:] #Only smallest image dimensions matter here

    #Pools are big, use a data generator
    pool_prep = ImageDataGenerator(
        samplewise_center=batch_norm,
        samplewise_std_normalization=batch_norm)

    #Acquisition functions that require a generator to load data
    generator_params = {
        'dps':data,
        'classes':ds.nclasses,
        'dim':fix_dim,
        'batch_size':batch_size,
        'image_generator':pool_prep,
        'shuffle':True,
        'verbose':verbose}

    X,Y = data
    All_Dropout_Classes = np.zeros(shape=(X.shape[0],1))

    generator = ThreadedGenerator(**generator_params)

    pred_model = None
    smodel,pmodel = None,None
    if os.path.isfile(model.get_mgpu_weights_cache()):
        try:
            smodel,pmodel = model.build()
            pmodel.load_weights(model.get_mgpu_weights_cache())
            pred_model = pmodel
            if kwargs['config'].info:
                print("Model weights loaded from: {0}".format(model.get_mgpu_weights_cache()))                
        except ValueError:
            pred_model = load_model(model.get_model_cache())
            if kwargs['config'].info:
                print("Model loaded from: {0}".format(model.get_model_cache()))

    if pbar:
        l = tqdm(range(mc_dp), desc="MC Dropout",position=0)
    else:
        if kwargs['config'].info:
            print("Starting MC dropout sampling...")
        l = range(mc_dp)
                
    for d in l:
        if pbar:
            print("\n")
        proba = pred_model.predict_generator(generator,
                                                workers=3*kwargs['config'].cpu_count,
                                                max_queue_size=25*gpu_count,
                                                verbose=verbose)

        dropout_classes = proba.argmax(axis=-1)    
        dropout_classes = np.array([dropout_classes]).T
        All_Dropout_Classes = np.append(All_Dropout_Classes, dropout_classes, axis=1)

    Variation = np.zeros(shape=(X.shape[0]))

    for t in range(X.shape[0]):
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
    
