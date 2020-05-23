#!/usr/bin/env python3
#-*- coding: utf-8

import os,sys
import numpy as np
import importlib
import random

#Filter warnings
import warnings
warnings.filterwarnings('ignore')
    
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

        self.pool_x = None
        self.pool_y = None
        self.train_x = None
        self.train_y = None
        self.val_x = None
        self.val_y = None
        self.test_x = None
        self.test_y = None
        if self._config.spool > 0:
            self.superp_x = None
            self.superp_y = None
            self.acq_idx = None
            self.sample_idx = None
            self.pool_size = None


    def _balance_classes(self,X,Y):
        """
        Returns a tuple (X,Y) of balanced classes

        X and Y are lists
        """
        #Work with NP arrays
        if isinstance(X,list):
            X = np.asarray(X)
        if isinstance(Y,list):
            Y = np.asarray(Y)
            
        #Count the occurrences of each class
        unique,count = np.unique(Y,return_counts=True)

        #Extracts the positions of each class in the dataset
        class_members = {i:np.where(Y == i)[0] for i in unique}

        #Remount the lists
        nX,nY = ([],[])
        mcount = count.min()
        for c in unique:
            if count[c] > mcount:
                ids = np.random.choice(count[c],mcount,replace=False)
                nX.extend(X[class_members[c][ids]])
                nY.extend(Y[class_members[c][ids]])
            else:
                nX.extend(X[class_members[c]])
                nY.extend(Y[class_members[c]])

        #Reshufle all elements
        combined = list(zip(nX,nY))
        random.shuffle(combined)
        nX[:],nY[:] = zip(*combined)

        return nX,nY


    def _split_origins(self,x_data,t_idx):
        """
        Separates patches of a predefined number of WSIs to be used as test set
        """

        cache_m = CacheManager()
        if cache_m.checkFileExistence('testset.pik'):
            full_id,samples = cache_m.load('testset.pik')
            if not samples is None and self._config.info:
                print("[ALTrainer] Using cached TEST SET. This is DANGEROUS. Use the metadata correspondent to the set.")
            return full_id,samples
            
        wsis = set()

        for k in x_data:
            wsis.add(k.getOrigin())

        wsis = list(wsis)
        selected = set(random.choices(wsis,k=self._config.wsi_split))
        selected_idx = []

        if self._config.info:
            print("[ALTrainer] WSIs selected to provide test patches:\n{}".format("\n".join(selected)))
            
        for i in range(len(x_data)):
            if x_data[i].getOrigin() in selected:
                selected_idx.append(i)

        t_idx = min(len(selected_idx),t_idx)
        samples = np.random.choice(selected_idx,t_idx,replace=False)
        full_id = np.asarray(selected_idx,dtype=np.int32)
        cache_m.dump((full_id,samples),'testset.pik')
        
        return full_id,samples

    def _refresh_pool(self,r,name):
        if self.superp_x is None or self.superp_y is None:
            return None
        
        #Store every pool
        cache_m = CacheManager()
        fid = 'al-pool-{1}-r{0}.pik'.format(r,name)
        cache_m.registerFile(os.path.join(self._config.logdir,fid),fid)
        
        self.pool_x = None
        self.pool_y = None

        if self._config.info:
            print("[ALTrainer] Regenerating pool from superpool ({} patches available)".format(self.superp_x.shape[0]))
        if self._config.verbose > 0:
            print("[ALTrainer] Removed from super pool ({1}): {0}".format(self.acq_idx,self.acq_idx.shape[0]))
            
        self.superp_x = np.delete(self.superp_x,self.acq_idx)
        self.superp_y = np.delete(self.superp_y,self.acq_idx)
        if cache_m.checkFileExistence(fid):
            self.sample_idx,_ = cache_m.load(fid)
            if self._config.info:
                print("[ALTrainer] Loaded resampled pool from: {}".format(cache_m.fileLocation(fid)))
        else:
            self.sample_idx = np.random.choice(range(self.superp_x.shape[0]),self.pool_size,replace=False)
            cache_m.dump((self.sample_idx,name),fid)
            
        self.pool_x = self.superp_x[self.sample_idx]
        self.pool_y = self.superp_y[self.sample_idx]
        self.acq_idx = None
        self._ds.check_paths(self.pool_x,self._config.predst)
        
        if self._config.info:
            print("[ALTrainer] Pool regenerated: {}. Superpool size: {}".format(self.pool_y.shape[0],self.superp_x.shape[0])) 
    
    def configure_sets(self):
        """
        Creates the initial sets: training (X,Y); example pool; validation set; test set

        All sets are kept as NP arrays
        """

        #Test set is extracted from the last items of the full DS or from a test dir and is not changed for the whole run
        fX,fY = self._ds.load_metadata()
        tsp = self._config.split[-1:][0]
        t_idx = 0
        if tsp > 1.0:
            t_idx = int(tsp)
        else:
            t_idx = int(tsp * len(fX))

        #Configuration option that limits test set size
        t_idx = min(self._config.pred_size,t_idx) if self._config.pred_size > 0 else t_idx

        if self._config.testdir is None or not os.path.isdir(self._config.testdir):
            if self._config.wsi_split > 0:
                full_id,samples = self._split_origins(fX,t_idx)
                self.test_x = fX[samples]
                self.test_y = fY[samples]
                X = np.delete(fX,full_id)
                Y = np.delete(fY,full_id)
            else:
                self.test_x = fX[- t_idx:]
                self.test_y = fY[- t_idx:]
                X,Y = fX[:-t_idx],fY[:-t_idx]
            self._ds.check_paths(self.test_x,self._config.predst)
        else:
            x_test,y_test = self._ds.run_dir(self._config.testdir)
            t_idx = min(len(x_test),t_idx)
            samples = np.random.choice(range(len(x_test)),t_idx,replace=False)
            self.test_x = [x_test[s] for s in samples]
            self.test_y = [y_test[s] for s in samples]
            del(x_test)
            del(y_test)
            del(samples)
            X,Y = fX,fY

        del(fX)
        del(fY)
        
        #Use a sample of the metadata if so instructed
        if self._config.sample != 1.0:
            if self._config.spool > 0:
                self.superp_x = X
                self.superp_y = Y            
            X,Y,self.sample_idx = self._ds.sample_metadata(self._config.sample,data=(X,Y),pos_rt=self._config.pos_rt)
            self._ds.check_paths(X,self._config.predst)

        if self._config.balance:
            X,Y = self._balance_classes(X,Y)
            if self._config.info:
                print("[ALTrainer] Using a balanced initial dataset for AL ({} total elements).".format(len(X)))
        elif self._config.info:
            print("[ALTrainer] Using an UNBALANCED initial dataset for AL ({} total elements).".format(len(X)))

        if self._config.spool > 0:
            self.pool_size = X.shape[0]
            
        self.pool_x = np.asarray(X)
        self.pool_y = np.asarray(Y)
        del(X)
        del(Y)

        #Initial training set will be choosen at random from pool if a default is not provided
        cache_m = CacheManager()
        if self._config.load_train and not self._config.balance and cache_m.checkFileExistence('initial_train.pik'):
            train_idx = cache_m.load('initial_train.pik')
            if not train_idx is None and self._config.info:
                print("[ALTrainer] Using initial training set from cache. This is DANGEROUS. Use the metadata correspondent to the initial set.")
        else:
            if not self._config.load_train and self._config.balance and self._config.info:
                print("[ALTrainer] Dataset balancing and initial train set loading not possible at the same time.")
                
            train_idx = np.random.choice(len(self.pool_x),self._config.init_train,replace=False)
            cache_m.dump(train_idx,'initial_train.pik')

        #Validation element index definition
        val_samples = int((self._config.init_train*self._config.split[1])/self._config.split[0])
        val_samples = max(val_samples,100)
        val_idx = np.random.choice(self.pool_x.shape[0],val_samples,replace=False)

        self.train_x = self.pool_x[train_idx]
        self.train_y = self.pool_y[train_idx]
        
        #Initial validation set - keeps the same split ratio for train/val as defined in the configuration
        self.val_x = self.pool_x[val_idx]
        self.val_y = self.pool_y[val_idx]

        #Remove the selected items from pool
        remove = np.concatenate((train_idx,val_idx),axis=0)
        self.pool_x = np.delete(self.pool_x,remove)
        self.pool_y = np.delete(self.pool_y,remove)
        self.sample_idx = np.delete(self.sample_idx,remove)
        
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
        predictor = Predictor(self._config,keepImg=self._config.keepimg)
        function = None
        
        if not self._config.ac_function is None:
            acq = importlib.import_module('AL','AcquisitionFunctions')
            function = getattr(acq,self._config.ac_function)
        else:
            print("You should specify an acquisition function")
            sys.exit(Exitcodes.RUNTIME_ERROR)

        stime = None
        etime = None
        sw_thread = None
        end_train = False
                    
        for r in range(self._config.acquisition_steps):
            if self._config.info:
                print("[ALTrainer] Starting acquisition step {0}/{1}".format(r+1,self._config.acquisition_steps))
                stime = time.time()

            #Save current dataset and report partial result (requires multi load for reading)
            fid = 'al-metadata-{1}-r{0}.pik'.format(r,model.name)
            cache_m.registerFile(os.path.join(self._config.logdir,fid),fid)
            cache_m.dump(((self.train_x,self.train_y),(self.val_x,self.val_y),(self.test_x,self.test_y)),fid)
                
            sw_thread = self.train_model(model,(self.train_x,self.train_y),(self.val_x,self.val_y))            
                
            if r == (self._config.acquisition_steps - 1) or not self.acquire(function,model,acquisition=r,sw_thread=sw_thread):
                if self._config.info:
                    print("[ALTrainer] No more acquisitions are in order")
                end_train = True
                
            #Some models may take too long to save weights
            if not sw_thread is None and sw_thread.is_alive():
                if self._config.info:
                    print("[ALTrainer] Waiting for model weights...")
                sw_thread.join()
                    
            #Set load_full to false so dropout is disabled
            predictor.run(self.test_x,self.test_y,load_full=False,net_model=model)
            
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
        import gc

        #Regenerate pool if defined
        if self._config.spool > 0 and kwargs['acquisition'] > 0 and ((kwargs['acquisition'] + 1) % (self._config.spool)) == 0:
            self._refresh_pool(kwargs['acquisition'],model.name)
            
        #Clear some memory before acquisitions
        gc.collect()
        
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
            'verbose':self._config.verbose}

        generator = ThreadedGenerator(**generator_params)

        if self._config.gpu_count > 1:
            pred_model = model.parallel
        else:
            pred_model = model.single

        if self._config.verbose > 0:
            print("\nStarting acquisition...(pool size: {})".format(self.pool_x.shape[0]))

        if self._config.verbose > 1:
            print("Starting acquisition using model: {0}".format(hex(id(pred_model))))
        
        pooled_idx = function(pred_model,generator,self.pool_x.shape[0],**kwargs)

        if pooled_idx is None:
            if self._config.info:
                print("[ALTrainer] No indexes returned. Something is wrong.")
            sys.exit(1)
            
        #Store acquired patches indexes in pool set
        if self._config.spool > 0:
            if self.acq_idx is None:
                self.acq_idx = self.sample_idx[pooled_idx]
            else:
                self.acq_idx = np.concatenate((self.acq_idx,self.sample_idx[pooled_idx]),axis=0)
            self.sample_idx = np.delete(self.sample_idx,pooled_idx)
            print("Pooled indexes ({}): {}".format(pooled_idx.shape[0],pooled_idx))
            print("Sample_idx ({}): {}".format(self.sample_idx.shape[0],self.sample_idx))
 
        del(generator)
        
        self.train_x = np.concatenate((self.train_x,self.pool_x[pooled_idx]),axis=0)
        self.train_y = np.concatenate((self.train_y,self.pool_y[pooled_idx]),axis=0)
        self.pool_x = np.delete(self.pool_x,pooled_idx)
        self.pool_y = np.delete(self.pool_y,pooled_idx)

        return True
