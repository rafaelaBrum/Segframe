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


def km_uncert(bayesian_model,generator,data_size,**kwargs):
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
    sw_threads <thread Object>: if a thread object is passed, you must wait its conclusion before loading weights
    """
    from sklearn.cluster import KMeans
    import importlib
    import copy
    import time
    from datetime import timedelta
    from Utils import CacheManager

    cache_m = CacheManager()
    
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
    
    if 'acquisition' in kwargs:
        r = kwargs['acquisition']
    else:
        r = config.acquisition_steps
        
    if 'model' in kwargs:
        model = kwargs['model']
    else:
        print("[km_uncert] GenericModel is needed by km_uncert. Set model kw argument")
        return None

    ## UNCERTAINTY CALCULATION FIRST 
    #Any uncertainty function could be used
    n_config = copy.copy(config)
    n_config.acquire = data_size
    kwargs['config'] = n_config
    un_function = getattr(importlib.import_module('AL'),config.un_function)
    un_indexes = un_function(bayesian_model,generator,data_size,**kwargs)    

    #Models that take to long to save weights might not have finished
    if 'sw_thread' in kwargs:
        if kwargs['sw_thread'].is_alive():
            if config.info:
                print("[km_uncert] Waiting for model weights to become available...")
            kwargs['sw_thread'].join()
    elif config.info:
        print("Weights thread not available...trying to load weights")

    if not os.path.isfile(model.get_weights_cache()) and not os.path.isfile(model.get_mgpu_weights_cache()):
        if config.info:
            print("[km_uncert] No trained model or weights file found")
        return None
    
    if hasattr(model,'build_extractor'):
        single_m,parallel_m = model.build_extractor(training=False,feature=True)
    else:
        if config.info:
            print("[km_uncert] Model is not prepared to produce features. No feature extractor")
        return None

    #Model can be loaded from previous acquisition train of from a fixed final model
    if gpu_count > 1 and not parallel_m is None:
        pred_model = parallel_m
        if not config.ffeat is None and os.path.isfile(config.ffeat):
            pred_model.load_weights(config.ffeat,by_name=True)
            if config.info:
                print("Model weights loaded from: {0}".format(config.ffeat))
        else:
            pred_model.load_weights(model.get_mgpu_weights_cache(),by_name=True)
            if config.info:
                print("Model weights loaded from: {0}".format(model.get_mgpu_weights_cache()))
    else:
        pred_model = single_m
        if not config.ffeat is None and os.path.isfile(config.ffeat):
            pred_model.load_weights(config.ffeat,by_name=True)
            if config.info:
                print("Model weights loaded from: {0}".format(config.ffeat))
        else:
            pred_model.load_weights(model.get_weights_cache(),by_name=True)
            if config.info:
                print("Model weights loaded from: {0}".format(model.get_weights_cache()))

    if config.info:
        print("Starting feature extraction ({} batches)...".format(len(generator)))

    if config.recluster > 0 and acq > 0 and (acq % config.recluster) != 0:
        km,acquired = cache_m.load('clusters.pik')
        if config.info:
            print("[km_uncert] Loaded clusters from previous acquisition")
            #TODO: REMOVE
            print("Previous cluster size: {};\nAcquired: {}".format(km.labels_.shape,acquired.shape))
        km.labels_ = np.delete(km.labels_,acquired)
        
    else:
        #Extract features for all images in the pool
        features = pred_model.predict_generator(generator,
                                                workers=4*cpu_count,
                                                max_queue_size=100*gpu_count,
                                                verbose=0)
        features = features.reshape(features.shape[0],np.prod(features.shape[1:]))

        stime = None
        etime = None
        if config.verbose > 0:
            print("Done extraction...starting KMeans")
            stime = time.time()
        
        km = KMeans(n_clusters = clusters, init='k-means++',n_jobs=int(cpu_count/2)).fit(features)
        
        if config.verbose > 0:
            etime = time.time()
            td = timedelta(seconds=(etime-stime))
            print("KMeans took {}".format(td))

    un_clusters = {k:[] for k in range(config.clusters)}

    #Distributes items in clusters in descending order of uncertainty
    for iid in un_indexes:
        un_clusters[km.labels_[iid]].append(iid)

    #Save clusters
    if config.save_var:
        fid = 'al-clustermetadata-{1}-r{0}.pik'.format(acq,model.name)
        cache_m.registerFile(os.path.join(config.logdir,fid),fid)
        cache_m.dump((generator.returnDataAsArray(),un_clusters,un_indexes),fid)
        
    #If debug
    if config.debug:
        expected = generator.returnLabelsFromIndex()
        for k in range(len(un_clusters)):
            ind = np.asarray(un_clusters[k])
            print("Cluster {}, # of items: {}".format(k,ind.shape[0]))
            posa = np.ndarray(shape=(1,),dtype=np.int32)
            for ii in range(min(ind.shape[0],30)):
                if ii == 0:
                    posa[0] = np.where(un_indexes == ind[ii])[0]
                else:
                    posa = np.hstack((posa,np.where(un_indexes == ind[ii])[0]))
            print("Cluster {} first items positions in index array (at most 30): {}".format(k,posa))
            #Check % of items of each class in cluster k
            c_labels = expected[ind]
            unique,count = np.unique(c_labels,return_counts=True)
            l_count = dict(zip(unique,count))
            if len(unique) > 2:
                print("Cluster {} items:".format(k))
                print("\n".join(["label {0}: {1} items" .format(key,l_count[key]) for key in unique]))
            else:
                if c_labels.shape[0] == 1:
                    l_count[c_labels[0] ^ 1] = 0
                print("Cluster {3} labels: {0} are 0; {1} are 1;\n - {2:.2f} are positives".format(l_count[0],l_count[1],(l_count[1]/(l_count[0]+l_count[1])),k))            
            
    ac_count = 0
    acquired = []
    j = 0
    while ac_count < query:
        cln = (ac_count+j) % clusters
        q = un_clusters[cln]
        if len(q) > 0:
            acquired.append(q.pop(0))
            ac_count += 1
        else:
            if verbose > 0:
                print("[km_uncert] Cluster {} exausted, will try to acquire image from cluster {}".format(cln,(cln+1)%clusters))
            j += 1
            continue

    acquired = np.asarray(acquired)
    if config.recluster > 0:
        cache_m.dump((km,acquired),'clusters.pik')
    
    return acquired
    
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
