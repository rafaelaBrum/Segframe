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


def km_varratios(bayesian_model,generator,data_size,**kwargs):
    """
    Cluster in K centroids and extract N samples from each cluster, based on maximum bayesian_varratios
    uncertainty.

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
    from sklearn.cluster import KMeans
    
    if 'config' in kwargs:
        config = kwargs['config']
        gpu_count = config.gpu_count
        cpu_count = config.cpu_count
        verbose = config.verbose
        pbar = config.progressbar
        query = config.acquire
        clusters = config.clusters
    else:
        return None

    if 'model' in kwargs:
        model = kwargs['model']
    else:
        print("[km_varratios] GenericModel is needed by km_varratios. Set model kw argument")
        return None

    if os.path.isfile(model.get_weights_cache()):
        if hasattr(model,'build_extractor'):
            pred_model,_ = model.build_extractor(training=False,feature=True)
        else:
            if config.info:
                print("Model is not prepared to produce features. No feature extractor")
            return None
        pred_model.load_weights(model.get_weights_cache(),by_name=True)
        if config.info:
            print("Model weights loaded from: {0}".format(model.get_weights_cache()))
    else:
        if config.info:
            print("No trained model or weights file found")
        return None

    if config.info:
        print("Starting feature extraction...")
        
    #Extract features for all images in the pool
    features = pred_model.predict_generator(generator,
                                             workers=4*cpu_count,
                                             max_queue_size=100*gpu_count,
                                             verbose=0)
    print("Features array shape: {}".format(features.shape))
    feature = features.reshape(features.shape[0],np.prod(features.shape[1:]))
    print("Features array shape: {}".format(features.shape))

    km = KMeans(n_clusters = clusters, init='k-means++',n_jobs=cpu_count).fit(features)
    print("Labels of some of the points: {}".format(np.random.choice(range(km.shape[0]),100,replace=False)))
    
    #TODO:
    #3- run pred_model.predict (on all samples - check if it's running in parallel)
    #4- cluster the data
    #5- run normal varratios
    #6- extract config.acquire/cluster items from each cluster, in decreasing order of uncertainty
    
    
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
    from Utils import CacheManager
    cache_m = CacheManager()
    
    if 'config' in kwargs:
        config = kwargs['config']
        mc_dp = config.dropout_steps
        gpu_count = config.gpu_count
        cpu_count = config.cpu_count
        verbose = config.verbose
        pbar = config.progressbar
        query = config.acquire
        save_var = config.save_var
    else:
        return None        

    if 'acquisition' in kwargs:
        r = kwargs['acquisition']

    fidp = None
    if save_var:
        fid = 'al-uncertainty-{1}-r{0}.pik'.format(r,config.ac_function)
        cache_m.registerFile(os.path.join(config.logdir,fid),fid)
        if config.debug:
            fidp = 'al-probs-{1}-r{0}.pik'.format(r,config.ac_function)
            cache_m.registerFile(os.path.join(config.logdir,fidp),fidp)
        
    All_Dropout_Classes = np.zeros(shape=(data_size,1))

    if pbar:
        l = tqdm(range(mc_dp), desc="MC Dropout",position=0)
    else:
        if config.info:
            print("Starting MC dropout sampling...")
        l = range(mc_dp)

    #Keep probabilities for analysis
    all_probs = None
    if config.debug:
        all_probs = np.zeros(shape=(mc_dp,data_size,generator.classes))
        
    for d in l:
        if pbar:
            print("\n")
        elif config.info:
            print("Step {0}/{1}".format(d+1,mc_dp))
           
        #Keep verbosity in 0 to gain speed 
        proba = pred_model.predict_generator(generator,
                                                workers=4*cpu_count,
                                                max_queue_size=100*gpu_count,
                                                verbose=0)

        if config.debug:
            all_probs[d] = proba
            
        dropout_classes = proba.argmax(axis=-1)    
        dropout_classes = np.array([dropout_classes]).T
        All_Dropout_Classes = np.append(All_Dropout_Classes, dropout_classes, axis=1)

    if verbose > 0:
        print("All dropout {0}:".format(All_Dropout_Classes.shape))
        for i in np.random.choice(All_Dropout_Classes.shape[0],100,replace=False):
            print("Predictions for image ({0}): {1}".format(i,All_Dropout_Classes[i]))
    
    Variation = np.zeros(shape=(data_size))

    for t in range(data_size):
        L = np.array([0])
        for d_iter in range(mc_dp):
            L = np.append(L, All_Dropout_Classes[t, d_iter+1])
        Predicted_Class, Mode = mode(L[1:])
        v = np.array(  [1 - Mode/float(mc_dp)])
        Variation[t] = v
    
    if verbose > 1:
        print("Variation {0}:".format(data_size))
        for i in np.random.choice(data_size,100,replace=False):
            print("Variation for image ({0}): {1}".format(i,Variation[i]))
        
    a_1d = Variation.flatten()
    x_pool_index = a_1d.argsort()[-query:][::-1]

    if config.debug:
        s_expected = generator.returnLabelsFromIndex(x_pool_index)
        #After transposition shape will be (classes,items,mc_dp)
        s_probs = all_probs[:mc_dp,x_pool_index].T
        debug_acquisition(s_expected,s_probs,generator.classes,cache_m,config,fidp)
            
    if save_var:
        cache_m.dump((x_pool_index,a_1d),fid)
        
    if verbose > 0:
        #print("Selected item indexes: {0}".format(x_pool_index))
        print("Selected item's variation: {0}".format(a_1d[x_pool_index]))
        print("Maximum variation in pool: {0}".format(a_1d.max()))
    
    return x_pool_index

def bayesian_bald(pred_model,generator,data_size,**kwargs):
    """
    Calculation as defined in paper:
    Bayesian convolutional neural networks with Bernoulli approximate variational inference
    """
    from Utils import CacheManager
    cache_m = CacheManager()

    if 'config' in kwargs:
        config = kwargs['config']
        mc_dp = config.dropout_steps
        gpu_count = config.gpu_count
        cpu_count = config.cpu_count
        verbose = config.verbose
        pbar = config.progressbar
        query = config.acquire
        save_var = config.save_var
    else:
        return None

    if 'acquisition' in kwargs:
        r = kwargs['acquisition']

    if save_var:
        fid = 'al-uncertainty-{1}-r{0}.pik'.format(r,config.ac_function)
        cache_m.registerFile(os.path.join(config.logdir,fid),fid)

    All_Entropy_Dropout = np.zeros(shape=data_size)
    score_All = np.zeros(shape=(data_size, generator.classes))
    
    if pbar:
        l = tqdm(range(mc_dp), desc="MC Dropout",position=0)
    else:
        if config.info:
            print("Starting MC dropout sampling...")
        l = range(mc_dp)
                
    for d in l:
        if pbar:
            print("\n")
        elif config.info:
            print("Step {0}/{1}".format(d+1,mc_dp))

        dropout_score = pred_model.predict_generator(generator,
                                                        workers=4*cpu_count,
                                                        max_queue_size=100*gpu_count,
                                                        verbose=0)
        #computing G_X
        score_All = score_All + dropout_score

        #computing F_X
        dropout_score_log = np.log2(dropout_score)
        Entropy_Compute = - np.multiply(dropout_score, dropout_score_log)
        Entropy_Per_Dropout = np.sum(Entropy_Compute, axis=1)
        
        All_Entropy_Dropout = All_Entropy_Dropout + Entropy_Per_Dropout 


    Avg_Pi = np.divide(score_All, mc_dp)
    Log_Avg_Pi = np.log2(Avg_Pi)
    Entropy_Avg_Pi = - np.multiply(Avg_Pi, Log_Avg_Pi)
    Entropy_Average_Pi = np.sum(Entropy_Avg_Pi, axis=1)

    G_X = Entropy_Average_Pi
    
    Average_Entropy = np.divide(All_Entropy_Dropout, mc_dp)

    F_X = Average_Entropy

    U_X = G_X - F_X

    # THIS FINDS THE MINIMUM INDEX 
    # a_1d = U_X.flatten()
    # x_pool_index = a_1d.argsort()[-Queries:]

    a_1d = U_X.flatten()
    x_pool_index = a_1d.argsort()[-query:][::-1]    

    if save_var:
        cache_m.dump((x_pool_index,a_1d),fid)
        
    if verbose > 0:
        #print("Selected item indexes: {0}".format(x_pool_index))
        print("Selected item's average entropy: {0}".format(a_1d[x_pool_index]))
        print("Maximum entropy in pool: {0}".format(a_1d.max()))
    
    return x_pool_index
    
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
