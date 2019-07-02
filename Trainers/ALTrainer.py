#!/usr/bin/env python3
#-*- coding: utf-8

import os,sys
import numpy as np
import importlib
from keras.preprocessing.image import ImageDataGenerator

#Local
from .GenericTrainer import Trainer
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
    trainer = ActiveLearningTrainer(config)
    trainer.run()
    
class ActiveLearningTrainer(Trainer):
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

        self.X = None
        self.Y = None
        self.pool_x = None
        self.pool_y = None
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None
        
    def configure_sets(self):
        """
        Creates the initial sets: training (X,Y); example pool; validation set; test set

        Except for self.X and self.Y all others are NP arrays!
        """
        self.X,self.Y = self._ds.load_metadata()

        #Test set is extracted from the last items and is not changed for the whole run
        t_idx = int(self._config.split[-1:][0] * len(self.X))
        self.test_x = self.X[- t_idx:]
        self.text_y = self.Y[- t_idx:]

        self.pool_x = self.X[:-t_idx]
        self.pool_y = self.Y[:-t_idx]

        #Initial training set will be choosen at random from pool
        train_idx = np.random.choice(len(self.pool_x),self._config.init_train,replace=False)
        pool_ar_x = np.asarray(self.pool_x)
        pool_ar_y = np.asarray(self.pool_y)
        self.train_x = pool_ar_x[train_idx]
        self.train_y = pool_ar_y[train_idx]

        #Remove choosen elements from the pool
        self.pool_x = np.delete(pool_ar_x,train_idx)
        self.pool_y = np.delete(pool_ar_y,train_idx)
        del(pool_ar_x)
        del(pool_ar_y)
        
        #Initial validation set - keeps the same split ratio for train/val as defined in the configuration
        val_samples = int((self._config.init_train*self._config.split[1])/self._config.split[0])
        val_idx = np.random.choice(self.pool_x.shape[0],val_samples,replace=False)
        self.val_x = self.pool_x[val_idx]
        self.val_y = self.pool_y[val_idx]
        self.pool_x = np.delete(self.pool_x,val_idx)
        self.pool_y = np.delete(self.pool_y,val_idx)
        
    def run(self):
        """
        Coordenates the AL process
        """
        #Loaded CNN model and Datasource
        model = self.load_modules()
        self._rex = self._rex.format(model.name)
        #Define initial sets
        self.configure_sets()
        #AL components
        cache_m = CacheManager()
        predictor = Predictor(self._config,keepImg=True)
        function = None
        
        if not self._config.ac_function is None:
            acq = importlib.import_module('AL','AcquisitionFunctions')
            function = getattr(acq,self._config.ac_function)
        else:
            print("You should specify an acquisition function")
            sys.exit(Exitcodes.RUNTIME_ERROR)
            
        for r in range(self._config.acquisition_steps):
            if self._config.info:
                print("[ALTrainer] Starting acquisition step {0}".format(r))
        
            self.train_model(model,(self.train_x,self.train_y),(self.val_x,self.val_y))
            
            #Save current dataset and report partial result (requires multi load for reading)
            fid = 'al-metadata-{1}-r{0}.pik'.format(r,model.name)
            cache_m.registerFile(os.path.join(self._config.cache,fid),fid)
            cache_m.dump(((self.train_x,self.train_y),(self.val_x,self.val_y),(self.test_x,self.test_y)),fid)

            predictor.run(self.test_x,self.test_y)
            
            if not self.acquire(function,model=model,config=self._config):
                if self._config.info:
                    print("[ALTrainer] No more acquisitions are possible")
                break
            

    def acquire(self,function,**kwargs):
        """
        Adds items to training and validation sets, according to split ratio defined in configuration. 
        Test set is fixed in the begining.

        Returns True if acquisition was sucessful
        """

        #An acquisition function should return a NP array with the indexes of all items from the pool that 
        #should be inserted into training and validation sets

        if self.pool_x.shape[0] < self._config.acquire:
            return False

        #Let acquisition functions access datasource if needed
        if kwargs is None:
            kwargs = {'ds':self._ds}
        else:
            kwargs['ds'] = self._ds
            
        pooled_idx = function((self.pool_x,self.pool_y),self._config.acquire,kwargs)
        self.train_x = np.concatenate((self.train_x,self.pool_x[pooled_idx]),axis=0)
        self.train_y = np.concatenate((self.train_y,self.pool_y[pooled_idx]),axis=0)
        self.pool_x = np.delete(self.pool_x,pooled_idx)
        self.pool_y = np.delete(self.pool_y,pooled_idx)

        return True
