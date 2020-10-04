#!/usr/bin/env python3
#-*- coding: utf-8

import numpy as np
import os

__doc__ = """
Utility functions for acquisition functions and independent functions
"""

def load_model_weights(config,model,single_m,parallel_m,sw_thread=None):

    npfile = False
    checkpath = None

    if not sw_thread is None:
        last_thread = None
        if isinstance(sw_thread,list):
            last_thread = sw_thread[-1]
        else:
            last_thread = sw_thread
        if config.ffeat is None and last_thread.is_alive():
            if config.info:
                print("[load_model_weights] Waiting for model weights to become available...")
            last_thread.join()
    elif config.info:
        print("[load_model_weights] Weights thread not available...trying to load weights")    
    
    if hasattr(model,'get_npweights_cache'):
        checkpath = model.get_npweights_cache(add_ext=True)
        spath = checkpath
        npfile = True
        
    if npfile and not os.path.isfile(checkpath):
        spath = model.get_weights_cache()
        npfile = False
            
            
    #Model can be loaded from previous acquisition train or from a fixed final model
    if config.gpu_count > 1 and not parallel_m is None:
        if hasattr(model,'get_npmgpu_weights_cache'):
            checkpath = model.get_npmgpu_weights_cache(add_ext=True)
            ppath = checkpath
            npfile = True
        
        if npfile and not os.path.isfile(checkpath):
            ppath = model.get_mgpu_weights_cache()
            npfile = False
            
        pred_model = parallel_m
        if not config.ffeat is None and os.path.isfile(config.ffeat):
            pred_model.load_weights(config.ffeat,by_name=True)
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(config.ffeat))
        elif npfile:
            pred_model.set_weights(np.load(ppath,allow_pickle=True))
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(ppath))
        else:
            pred_model.load_weights(ppath,by_name=True)
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(ppath))
    else:
        pred_model = single_m
        if not config.ffeat is None and os.path.isfile(config.ffeat):
            pred_model.load_weights(config.ffeat,by_name=True)
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(config.ffeat))
        elif npfile:
            pred_model.set_weights(np.load(spath,allow_pickle=True))
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(spath))                
        else:
            pred_model.load_weights(spath,by_name=True)
            if config.info and not config.progressbar:
                print("Model weights loaded from: {0}".format(spath))

    return pred_model

def random_sample(pred_model,generator,data_size,**kwargs):
    """
    Returns a random list of indexes from the given dataset
    """
    if 'config' in kwargs:
        k = kwargs['config'].acquire
    else:
        return None
    
    return np.random.choice(range(data_size),k,replace=False)

def oracle_sample(pred_model,generator,data_size,**kwargs):
    """
    Returns the indexes of images that the current network missclassified with high probability.
    """

    if 'config' in kwargs:
        acquire = kwargs['config'].acquire
        cpu_count = kwargs['config'].cpu_count
        gpu_count = kwargs['config'].gpu_count
    else:
        return None

    if kwargs['config'].info:
        print("Oracle prediction starting...")
        
    #Keep verbosity in 0 to gain speed
    proba = pred_model.predict_generator(generator,
                                             workers=4*cpu_count,
                                             max_queue_size=100*gpu_count,
                                             verbose=0)
            
    pred_classes = proba.argmax(axis=-1)    
    expected = generator.returnLabelsFromIndex()
    miss = np.where(pred_classes != expected)[0]
    miss_prob = np.zeros(shape=expected.shape)
    for k in range(miss.shape[0]):
        miss_prob[miss[k]] = proba[miss[k]][pred_classes[miss[k]]]

    x_pool_idx = np.argsort(miss_prob)[-acquire:]
    
    if kwargs['config'].verbose > 0:
        print('Misses ({}): {}'.format(miss.shape[0]/expected.shape[0],miss))
        print("Probabilities for selected items:\n {}".format(miss_prob[x_pool_idx]))
        print("Selected item's prediction/true label:\n Prediction: {}\n True label: {}".format(pred_classes[x_pool_idx],
                                                                                                   expected[x_pool_idx]))

    return x_pool_idx


def debug_acquisition(s_expected,s_probs,classes,cache_m,config,fidp):
    from Utils import PrintConfusionMatrix
    
    if config.verbose > 0:
        r_class = np.random.randint(0,classes)
        print("Selected item's probabilities for class {2} ({1}): {0}".format(s_probs[r_class],s_probs.shape,r_class))
        prob_mean = np.mean(np.mean(s_probs,axis=-1),axis=-1)
        print("\n".join(["Selected item's mean probabilities for class {}:{}".format(k,prob_mean[k]) for k in range(prob_mean.shape[0])]))
        
    s_pred_all = s_probs[:,:].argmax(axis=0)

    if config.verbose > 0:
        print("Votes: {}".format(s_pred_all))
    #s_pred holds the predictions for each item after a vote
    s_pred = np.asarray([np.bincount(s_pred_all[i]).argmax(axis=0) for i in range(0,s_pred_all.shape[0])])
    if config.verbose > 0:
        print("Classification after vote: {}".format(s_pred))
    PrintConfusionMatrix(s_pred,s_expected,classes,config,"Selected images (AL)")
    if config.save_var:
        cache_m.dump((s_expected,s_probs),fidp)
