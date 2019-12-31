#!/usr/bin/env python3
#-*- coding: utf-8

import numpy as np
import os


__doc__ = """
Utility functions for acquisition functions
"""

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
