#!/usr/bin/env python3
#-*- coding: utf-8

import numpy as np
from scipy.stats import mode

__doc__ = """
All acquisition functions should receive:
1 - numpy array of items
2 - numpy array of labels
3 - number of items to query
4 - keyword arguments specific for each function (if needed)

Returns: numpy array of element indexes
"""

def _predict_classes(model,generator,batch_size,verbose=1):
        proba = model.predict_generator(generator,
                                        batch_size=batch_size,
                                        max_queue_size=20,
                                        verbose=verbose)
        return proba.argmax(axis=-1)

def bayesian_varratios(generator,query,kwargs):
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
    All_Dropout_Classes = np.zeros(shape=(pool_x.shape[0],1))
    for d in range(mc_dp):
        dropout_classes = _predict_classes(model.single,generator,batch_size=batch_size, verbose=1)
        dropout_classes = np.array([dropout_classes]).T
        All_Dropout_Classes = np.append(All_Dropout_Classes, dropout_classes, axis=1)

    Variation = np.zeros(shape=(pool_x.shape[0]))

    for t in range(pool_x.shape[0]):
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
    
