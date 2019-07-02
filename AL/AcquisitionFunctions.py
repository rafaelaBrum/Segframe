#!/usr/bin/env python3
#-*- coding: utf-8

import numpy as np
from scipy.stats import mode

from Utils import multigpu_run

__doc__ = """
All acquisition functions should receive:
1 - numpy array of items
2 - numpy array of labels
3 - number of items to query
4 - keyword arguments specific for each function (if needed)

Returns: numpy array of element indexes
"""

def _predict_classes(data,model,generator_params,verbose=1):
    generator_params['dps']=data
    generator = ThreadedGenerator(**generator_params)

    proba = model.predict_generator(generator,
                                        max_queue_size=40,
                                        verbose=verbose)
    return proba.argmax(axis=-1)

def bayesian_varratios(data,query,kwargs):
    """
    Calculation as defined in paper:
    Bayesian convolutional neural networks with Bernoulli approximate variational inference

    @param data <tuple>: X,Y as numpy arrays
    """
    from Trainers import ThreadedGenerator
    from keras.preprocessing.image import ImageDataGenerator
    from keras import backend as K
    
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
    generator_params = {'classes':ds.nclasses,
                            'dim':fix_dim,
                            'batch_size':batch_size,
                            'image_generator':pool_prep,
                            'shuffle':True,
                            'verbose':verbose}

    X,Y = data
    All_Dropout_Classes = np.zeros(shape=(X.shape[0],1))
    for d in range(mc_dp):
        if gpu_count <= 1:
            dropout_classes = _predict_classes(data,model.single,generator_params, verbose=1)
        else:
            #Closes parent tf.Session for multiprocess run
            sess = K.get_session()
            sess.close()
            dropout_classes = multigpu_run(_predict_classes,
                                               (model.single,generator_params,verbose),data,
                                               gpu_count,pbar,txt_label='Running MC Dropout..',
                                               verbose=verbose)
            #Restores session after child process terminates
            sess = tf.Session()
            K.set_session(sess)
            
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
    
