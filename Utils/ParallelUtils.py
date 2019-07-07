#!/usr/bin/env python3
#-*- coding: utf-8

import sys
import os
import numpy as np
import multiprocessing
import concurrent.futures

from tqdm import tqdm

def multiprocess_run(exec_function,exec_params,data,cpu_count,pbar,step_size,output_dim=1,txt_label='',verbose=False):
    """
    Runs exec_function in a process pool. This function should receive parameters as follows:
    (iterable_data,param2,param3,...), where paramN is inside exec_params 

    @param exec_function <function>
    @param exec_params <tuple>
    @param data <iterable>
    @param cpu_count <int>: use this number of cores
    @param pbar <boolean>: user progress bars
    @param step_size <int>: size of the iterable that exec_function will receive
    @param output_dim <int>: exec_function produces how many sets of results?
    """
    
    # Perform extractions of frames in parallel and in steps
    step = int(len(data) / step_size) + (len(data)%step_size>0)
    datapoints_db = [[] for i in range(output_dim)]
    semaphores = []

    process_counter = 0
    pool = multiprocessing.Pool(processes=cpu_count,maxtasksperchild=50,
                                    initializer=tqdm.set_lock, initargs=(multiprocessing.RLock(),))

    if pbar:
        l = tqdm(desc="Processing {0}...".format(txt_label),total=step,position=0)
   
    datapoints = np.asarray(data)
    for i in range(step):
        # get a subset of datapoints
        end_idx = step_size
        
        if end_idx > len(data):
            end_idx = len(data)
        
        cur_datapoints = datapoints[:end_idx]

        if pbar:
            semaphores.append(pool.apply_async(exec_function,
                                args=(cur_datapoints,) + exec_params,
                                callback=lambda x: l.update(1)))
        else:
            semaphores.append(pool.apply_async(exec_function,
                                args=(cur_datapoints,) + exec_params))
        
        datapoints = np.delete(datapoints,np.s_[:end_idx],axis=0)

        if pbar:
            if process_counter == cpu_count+1:
                semaphores[process_counter].wait()
                process_counter = 0
            else:
                process_counter += 1

        #datapoints = np.delete(datapoints,np.s_[i*step_size : end_idx],axis=0)        
        #del cur_datapoints    
            
    for i in range(len(semaphores)):
        res = semaphores[i].get()
        for k in range(output_dim):
            datapoints_db[k].extend(res[k])
        if not pbar and verbose > 0:
            print("[{2}] Done transformations (step {0}/{1})".format(i,len(semaphores)-1,txt_label))

    if pbar:
        l.close()
        print("\n"*cpu_count)

    #Free all possible memory
    pool.close()
    pool.join()

    del datapoints
    
    # remove None points
    return tuple(filter(lambda x: not x is None, datapoints_db))
    

def multigpu_run(exec_function,exec_params,model,data,gpu_count,pbar,step_size=None,output_dim=1,txt_label='',verbose=False):
    """
    Runs exec_function in a process pool. This function should receive parameters as follows:
    (iterable_data,param2,param3,...), where paramN is inside exec_params
    Function should be something that deals with a CNN in Keras: train, predict, etc.

    @param exec_function <function>
    @param exec_params <tuple>
    @param model <GenericModel>: generic model to be loaded
    @param data <tuple>: (X,Y) as numpy arrays each
    @param gpu_count <int>: use this number of cores
    @param pbar <boolean>: user progress bars
    @param step_size <int>: size of the iterable that exec_function will receive
    @param output_dim <int>: exec_function produces how many sets of results?
    """
    
    #def _initializer(q,processes):

        #initialize tensorflow session
    #    gpu_options = None
    #    if not q is None:
    #        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    #        gpu_options.allow_growth = True
    #        gpu_options.Experimental.use_unified_memory = False
    #        gpu_options.visible_device_list = "{0}".format(q.get())

        #s_config = tf.ConfigProto(        
    #    sess = tf.Session(config=tf.ConfigProto(        
    #        device_count={"CPU":processes,"GPU":1},
    #        intra_op_parallelism_threads=processes, 
    #        inter_op_parallelism_threads=processes,
    #        log_device_placement=False,
    #        gpu_options=gpu_options
    #        ))
        #sess.config = s_config
    #    K.set_session(sess)
    #    print("[multigpu_run] DONE INITIALIER")
    
    from keras import backend as K
    import tensorflow as tf
    from keras.models import load_model

    def thread_worker(q,graph,*args):

        gpu = q.get()
        tfdevice = '/device:GPU:{0}'.format(gpu)

        sess = tf.Session(graph=graph)
        ses_config = tf.ConfigProto(
                device_count={"CPU":3,"GPU":1},
                intra_op_parallelism_threads=3,
                inter_op_parallelism_threads=3,
                log_device_placement=False
                )
        sess.config = ses_config

        with sess:
            with tf.device(tfdevice):
                if verbose:
                    print("Runing on GPU {0}".format(gpu))
                r_val = exec_function(*args)
        return r_val

    data_size = data[0].shape[0]
    if gpu_count > 1:
        step_size = int(data_size / gpu_count)
        step = gpu_count
    else:
        step_size = data_size
        step = 1
            
    #GPU assignment for each process is defined by the queue
    device_queue = None
    if gpu_count > 0:
        device_queue = multiprocessing.Queue()
        for dev in range(0,step):
            device_queue.put(dev%gpu_count)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=gpu_count)
    
    #Clears session for the threads
    #K.clear_session()
    pmodel = None
    if os.path.isfile(model.get_model_cache()):
        try:
            pmodel = load_model(model.get_model_cache())
            if verbose:
                print("[MC Dropout] Model loaded from: {0}".format(model.get_model_cache()))
        except ValueError:
            pmodel,_ = model.build()
            pmodel.load_weights(model.get_weights_cache())
    else:
        return None

    graph = tf.get_default_graph()
    
    datapoints_db = []
    semaphores = []
    
    if pbar:
        l = tqdm(desc=txt_label,total=step,position=0)
    
    for i in range(step):
        # get a subset of datapoints
        end_idx = (i+1)*step_size        
        
        if (end_idx < data_size) and i == step-1:
            end_idx = data_size
        
        cur_datapoints = (data[0][i*step_size : end_idx],data[1][i*step_size : end_idx])
        #cur_datapoints = datapoints[:end_idx]

        args = (cur_datapoints,pmodel) + exec_params
        semaphores.append(executor.submit(thread_worker,device_queue,graph,*args))
            
    for i in range(len(semaphores)):
        datapoints_db.extend(semaphores[i].result())
        if not pbar and verbose > 0:
            print("Done predicting (group {0}/{1})".format(i,len(semaphores)-1))
            
    
    return datapoints_db
