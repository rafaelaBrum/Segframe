#!/usr/bin/env python3
#-*- coding: utf-8

import os,sys
import numpy as np
import importlib
import random
from multiprocessing import Pool, Queue

from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import tensorflow as tf

#Local
from .ALTrainer import ActiveLearningTrainer
from .Predictions import Predictor

#Module
from Utils import Exitcodes,CacheManager

def run_training(config,locations=None):
    """
    Main training function, to work as a new process
    """
    if config.info:
        print("Starting active learning process....")

    if not locations is None:
        cache_m = CacheManager(locations=locations)
    trainer = EnsembleALTrainer(config)
    trainer.run()

class EnsembleALTrainer(ActiveLearningTrainer):
    """
    Implements the structure of active learning:
    - Uses a selection function to acquire new training points;
    - Manages the training/validation/test sets

    Methods that should be overwriten by specific AL strategies:
    - 
    """

    def __init__(self,config):
        """
        @param config <argparse>: A configuration object
        """
        super().__init__(config)
        self.tower = None

    def _initializer(self,q,processes):

        #initialize tensorflow session
        gpu_options = None
        self._allocated_gpus = 0
        if not q is None:
            gpus = q.get()
            self._allocated_gpus = len(gpus)
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
            gpu_options.allow_growth = True
            gpu_options.Experimental.use_unified_memory = False
            gpu_options.visible_device_list = ",".join([str(g) for g in gpus])

        sess = tf.Session(config=tf.ConfigProto(
            device_count={"CPU":processes,"GPU":0 if q is None else 1},
            intra_op_parallelism_threads=3, 
            inter_op_parallelism_threads=3,
            log_device_placement=False,
            gpu_options=gpu_options
            ))
        #sess.config = config
        K.set_session(sess)

    def _child_run(self,model,m,train,val,ppgpus):

        if hasattr(model,'register_ensemble'):
            model.register_ensemble(m)
        else:
            print("Model not ready for ensembling. Implement register_ensemble method")
            raise AttributeError

        sw_thread = self.train_model(model,train,val,set_session=False,verbose=2,summary=False,clear_sess=True,allocated_gpus=ppgpus)
        if sw_thread.is_alive():
            sw_thread.join()
        return True
        
    def run(self):
        """
        Coordenates the AL process
        """
        from keras import backend as K
        import time
        from datetime import timedelta
        
        #Loaded CNN model and Datasource
        model = self.load_modules()
        self._rex = self._rex.format(model.name)
        #Define initial sets
        self.configure_sets()
        #AL components
        cache_m = CacheManager()
        predictor = Predictor(self._config,keepImg=True,build_ensemble=True)
        function = None
        
        if not self._config.ac_function is None:
            acq = importlib.import_module('AL','AcquisitionFunctions')
            function = getattr(acq,self._config.ac_function)
        else:
            print("You should specify an acquisition function")
            sys.exit(Exitcodes.RUNTIME_ERROR)

        stime = None
        etime = None
        end_train = False
        for r in range(self._config.acquisition_steps):
            if self._config.info:
                print("[ALTrainer] Starting acquisition step {0}/{1}".format(r+1,self._config.acquisition_steps))
                stime = time.time()

            #Save current dataset and report partial result (requires multi load for reading)
            fid = 'al-metadata-{1}-r{0}.pik'.format(r,model.name)
            cache_m.registerFile(os.path.join(self._config.logdir,fid),fid)
            cache_m.dump(((self.train_x,self.train_y),(self.val_x,self.val_y),(self.test_x,self.test_y)),fid)

            #Define GPU allocations
            device_queue = None
            ppgpus = 0
            if self._config.gpu_count > 0:
                device_queue = Queue()
                if (self._config.gpu_count == self._config.emodels) or (self._config.gpu_count % self._config.emodels):
                    ppgpus = 1
                    if self._config.gpu_count > self._config.emodels:
                        print("{} ensemble models and {} GPUs available is a waste. Choose carefully".format(self._config.emodels,self._config.gpu_count))
                        sys.exit(1)
                    for dev in range(self._config.emodels):
                        slot = (dev%self._config.gpu_count,)
                        device_queue.put(slot)
                else:
                    gi = 0
                    ppgpus = self._config.gpu_count // self._config.emodels
                    for dev in range(self._config.emodels):
                        slot = tuple(range(gi,gi+ppgpus))
                        device_queue.put(slot)
                        gi += ppgpus

            processes = self._config.gpu_count if ppgpus == 1 else ppgpus if ppgpus > 1 else self._config.cpu_count
            pool = Pool(processes=processes,initializer=self._initializer,
                            initargs=(device_queue,self._config.cpu_count),maxtasksperchild=self._config.emodels)

            #Schedule training for every ensemble model
            results = []
            ccount = 0
            allocations = [None for _ in range(processes)]
            for m in range(self._config.emodels):
                if ccount < processes:
                    if self._config.info:
                        print("[EnsembleTrainer] Starting ensemble model {} training..".format(m))
                    args=(model,m,(self.train_x,self.train_y),(self.val_x,self.val_y),ppgpus)
                    asr = pool.apply_async(self._child_run,args=args)
                    results.append(asr)
                    allocations[ccount] = asr
                    ccount += 1
                else:
                    k = 0
                    while True:
                        k = k % len(allocations)
                        if allocations[k].ready():
                            if self._config.verbose > 0:
                                print("[EnsembleTrainer] Ensemble model {}, will begin training.".format(m))
                            args=(model,m,(self.train_x,self.train_y),(self.val_x,self.val_y),ppgpus)
                            asr = pool.apply_async(self._child_run,args=args)
                            results.append(asr)
                            allocations[k] = asr
                            break
                        else:
                            if self._config.verbose > 1:
                                print("[EnsembleTrainer] Ensemble model {}, waiting for resources to became available.".format(m))
                            allocations[k].wait(60)
                            k += 1
                    
            pool.close()
            pool.join()
            
            if r == (self._config.acquisition_steps - 1) or not self.acquire(function,model,acquisition=r,sw_thread=None):
                if self._config.info:
                    print("[ALTrainer] No more acquisitions are in order")
                end_train = True
                    
            #Set load_full to false so dropout is disabled
            predictor.run(self.test_x,self.test_y,load_full=False)
            
            #Attempt to free GPU memory
            K.clear_session()
            
            if self._config.info:
                etime = time.time()
                td = timedelta(seconds=(etime-stime))
                print("Acquisition step took: {0}".format(td))
                
            if end_train:
                return None

    def acquire(self,function,model,**kwargs):
        """
        Adds items to training and validation sets, according to split ratio defined in configuration. 
        Test set is fixed in the begining.

        Returns True if acquisition was sucessful
        """
        from Trainers import ThreadedGenerator
        #An acquisition function should return a NP array with the indexes of all items from the pool that 
        #should be inserted into training and validation sets
        if self.pool_x.shape[0] < self._config.acquire:
            return False

        if kwargs is None:
            kwargs = {}

        kwargs['config'] = self._config

        #Some acquisition functions may need access to GenericModel
        kwargs['model'] = model
        
        if not self._config.tdim is None:
            fix_dim = self._config.tdim
        else:
            fix_dim = self._ds.get_dataset_dimensions()[0][1:] #Only smallest image dimensions matter here

        #Pools are big, use a data generator
        pool_prep = ImageDataGenerator(
            samplewise_center=self._config.batch_norm,
            samplewise_std_normalization=self._config.batch_norm)

        #Acquisition functions that require a generator to load data
        generator_params = {
            'dps':(self.pool_x,self.pool_y),
            'classes':self._ds.nclasses,
            'dim':fix_dim,
            'batch_size':self._config.gpu_count * self._config.batch_size if self._config.gpu_count > 0 else self._config.batch_size,
            'image_generator':pool_prep,
            'shuffle':False, #DO NOT SET TRUE!
            'verbose':self._config.verbose,
            'input_n':self._config.emodels}

        generator = ThreadedGenerator(**generator_params)

        if self._config.verbose > 0:
            print("\nStarting acquisition...(pool size: {})".format(self.pool_x.shape[0]))

        if self._config.verbose > 1:
            print("Starting acquisition using model: {0}".format(hex(id(pred_model))))
        
        pooled_idx = function(None,generator,self.pool_x.shape[0],**kwargs)
        if pooled_idx is None:
            if self._config.info:
                print("[ALTrainer] No indexes returned. Something is wrong.")
            sys.exit(1)
        self.train_x = np.concatenate((self.train_x,self.pool_x[pooled_idx]),axis=0)
        self.train_y = np.concatenate((self.train_y,self.pool_y[pooled_idx]),axis=0)
        self.pool_x = np.delete(self.pool_x,pooled_idx)
        self.pool_y = np.delete(self.pool_y,pooled_idx)

        return True        
