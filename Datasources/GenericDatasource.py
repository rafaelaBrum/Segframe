#!/usr/bin/env python3
#-*- coding: utf-8

from abc import ABC,abstractmethod
from tqdm import tqdm

import concurrent.futures
import numpy as np
import os
import random

from Utils import CacheManager,multiprocess_run

class GenericDS(ABC):
    """
    Generic class for data feeders used to provide training points to Neural Nets.
    """
    def __init__(self,data_path,keepImg=False,config=None,name='Generic'):
        self.path = None
        if isinstance(data_path,str) and os.path.isdir(data_path):
            self.path = data_path
        else:
            raise ValueError("[GenericImage] Path does not correspond to a file ({0}).".format(data_path))

        self.X = None
        self.Y = None
        self.name = name
        self.multi_dir = True
        self._cache = CacheManager()
        self._keep = keepImg
        self._cpu_count = config.cpu_count if not config is None else 1
        self._verbose = config.verbose if not config is None else 0
        self._pbar = config.progressbar if not config is None else False
        self._config = config


    @abstractmethod
    def _load_metadata_from_dir(self,d):
        pass

    def get_dataset_dimensions(self,X = None):
        """
        Returns the dimensions of the images in the dataset. It's possible to have different image dimensions.
        WARNING: big datasets will take forever to run. For now, checks a sample of the images.
        TODO: Reimplement this function to be fully parallel (threads in case).

        Return: SORTED list of tuples (# samples,width,height,channels)
        """

        cache_m = CacheManager()
        reload_data = False
        if cache_m.checkFileExistence('data_dims.pik'):
            try:
                dims,name = cache_m.load('data_dims.pik')
            except ValueError:
                reload_data = True
            if name != self.name:
                reload_data = True
        else:
            reload_data = True
                
        if reload_data:
            dims = set()
            if X is None and self.X is None:
                return None
            elif X is None:
                X = self.X
        
            samples = len(X)            
            if self._config.info:
                print("Checking a sample of dataset images for different dimensions...")

            s_number = int(0.02*samples)
            upper_limit = 5000 if s_number > 5000 else s_number
            for seg in random.sample(X,upper_limit):
                dims.add((samples,) + seg.getImgDim())
            cache_m.dump((dims,self.name),'data_dims.pik')

        l = list(dims)
        l.sort()
        return l

    def _run_multiprocess(self,data):
        """
        This method should not be called directly. It's intended
        only for multiprocess metadata loading.
        """
        X,Y = ([],[])
        for item in data:
            t_x,t_y = self._load_metadata_from_dir(item)
            X.extend(t_x)
            Y.extend(t_y)

        return (X,Y)

    def _split_data(self,split,X,Y):
        """
        Split data in at most N sets. Returns a tuple (set1,set2,set3,setN) with the divided
        data
        """
        if sum(split) == 1.0:
            it_count = 0
            split_data = []
            start_idx = 0
            samples = len(X)
            for frac in split:
                it_count = int(frac * samples)
                split_data.append((X[start_idx:start_idx+it_count],Y[start_idx:start_idx+it_count]))
                start_idx += it_count
            return split_data
                
        else:
            raise ValueError("[GenericDatasource] Spliting values have to equal 1.0")

    def _run_dir(self,path):

        dlist = []
        files = os.listdir(path)
        X,Y = ([],[])

        if self.multi_dir:
            for f in files:
                item = os.path.join(path,f)
                if os.path.isdir(item):
                    dlist.append(item)

            mdata = multiprocess_run(self._run_multiprocess,tuple(),dlist,
                                        self._cpu_count,self._pbar,
                                        step_size=1,output_dim=2,txt_label='directories',verbose=self._verbose)

        else:
            mdata = self._load_metadata_from_dir(self.path)

        X.extend(mdata[0]) #samples
        Y.extend(mdata[1]) #labels

        X,Y = self._shuffle(X,Y)
        return X,Y

    def _shuffle(self,X,Y):
        #Shuffle samples and labels maintaining relative order
        combined = list(zip(X,Y))
        random.shuffle(combined)
        X[:],Y[:] = zip(*combined)

        return X,Y
        
    def split_metadata(self,split,data=None):
        """
        Returns all metadata split into N sets, defined by the spliting tuples
        
        @param data <tuple>: (X,Y) if provided, split this sequence. Else, split full metadata
        """
        if data is None:
            return self._split_data(split,self.X,self.Y)
        elif len(data) == 2:
            return self._split_data(split,data[0],data[1])
        else:
            return None
    
    def load_metadata(self,metadata_file='metadata.pik'):
        """
        Iterates over data patches and creates an instance of a GenericImage subclass for each one
        Returns a tuples of lists (X,Y): X instances of GenericImage subclasses, Y labels;

        OBS: Dataset metadata is shuffled once here. Random sample generation is done during training.
        """

        X,Y = (None,None)
        reload_data = False
        reshuffle = False
        
        if self._cache.checkFileExistence('split_ratio.pik'):
            split = self._cache.load('split_ratio.pik')
            if self._config.split != split:
                #Dump old data
                reshuffle = True
                if not self.X is None or not self.Y is None:
                    del(self.X)
                    del(self.Y)
                    self.X = None
                    self.Y = None
                    
                if self._config.info:
                    print("Previous split ratio {} is different from requested one {}. Metadata will be reshuffled.".format(split,self._config.split))
                
        if self._cache.checkFileExistence(metadata_file) and not reload_data:
            try:
                X,Y,name = self._cache.load(metadata_file)
            except ValueError:
                name = ''
                reload_data = True
            if name != self.name:
                reload_data = True

            if not reload_data and not reshuffle and self._verbose > 0:
                print("[GenericDatasource] Loaded split data cache. Used previously defined splitting.")
        else:
            reload_data = True


        if reshuffle:
            X,Y = self._shuffle(X,Y)
            
        if reload_data:
            X,Y = self._run_dir(self.path)

        if reload_data or reshuffle:
            self._cache.dump((X,Y,self.name),metadata_file)
            self._cache.dump(tuple(self._config.split),'split_ratio.pik')
            
        self.X = X.copy()
        self.Y = Y.copy()
        return X,Y
    
    def load_data(self,split=None,keepImg=False,data=None):
        """
        Actually reads images and returns data ready for training
        Returns two tuples of NP arrays (X,Y): X data points, Y labels;

        @param split <tuple>: items are spliting fractions

        If a spliting ratio is provided, return a list of tuples of size at most 3:
        1 - Train;
        2 - Validation;
        3 - Test;
        
        @param keepImg <bool>: Keep image data in memory
        @param data <tuple>: metadata defining images to load. If not provided, full dataset is used.
        """

        if data is None and (self.X is None or self.Y is None):
            if self._verbose > 0:
                print("[GenericDatasource] Metadata not ready, loading...")
            self.load_metadata()

        #Which data to use?
        X,Y = None,None
        if data is None:
            X = self.X
            Y = self.Y
        else:
            X,Y = data
        
        if self._config.pred_size > 0:
            samples = self._config.pred_size
        else:    
            samples = len(X)
        y = np.array(Y[:samples], dtype=np.int32)
        if not self._config.tdim is None and len(self._config.tdim) == 2:
            img_dim = tuple(self._config.tdim) + (3,)
        else:
            dataset_dim = self.get_dataset_dimensions(X)[0]
            img_dim = dataset_dim[1:]
        X_data = np.zeros(shape=(samples,)+img_dim, dtype=np.float32)
        
        counter = 0
        futures = []

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=7)
        for i in range(samples):
            futures.append(executor.submit(X[i].readImage,keepImg,img_dim,self._verbose))

        if self._pbar:
            l = tqdm(desc="Reading images...",total=samples,position=0)
        elif self._config.info:
            print("Reading images...")
        
        #for future in concurrent.futures.as_completed(futures):
        for i in range(samples):
            X_data[i] = futures[i].result()
            if self._pbar:
                l.update(1)
            elif self._verbose > 0:
                print(".",end='')
            
        if self._pbar:
            l.close()
        elif self._verbose > 0:
            print('\n')

        if split is None:
            return (X_data,y)
        else:
            return self._split_data(split,X_data,y)

    def sample_metadata(self,k):
        """
        Produces a sample of the full metadata with k items. Returns a cached sample if one exists

        @param k <int>: total of samples
        @param k <float>: percentile of the whole dataset

        Return:
        - tuple (X,Y): X an Y have k elements
        """

        reload_data = False
        s_x,s_y = (None,None)
        if self._cache.checkFileExistence('sampled_metadata.pik'):
            try:
                s_x,s_y,name = self._cache.load('sampled_metadata.pik')
            except ValueError:
                name = ''
                reload_data = True
            if name != self.name:
                reload_data = True

            #Check if we have the desired number of items
            if k <= 1.0:
                k = int(k*len(self.X))
            else:
                k = int(k)
            if k != len(s_x):
                if self._config.info:
                    print("Saved samples are different from requested ({} x {}). Resampling...".format(k,len(s_x)))
                reload_data = True
                
            if not reload_data and self._verbose > 0:
                print("[GenericDatasource] Loaded split sampled data cache. Used previously defined splitting.")
        else:
            reload_data = True
        
        if reload_data and (self.X is None or self.Y is None):
            if self._config.verbose > 1:
                print("[GenericDatasource] Run load_metadata first!")
            return None
        
        if reload_data:
            if k <= 1.0:
                k = int(k*len(self.X))
            else:
                k = int(k)
            
            samples = np.random.choice(range(len(self.X)),k,replace=False)
            
            s_x = [self.X[s] for s in samples]
            s_y = [self.Y[s] for s in samples]

        #Save last generated sample
        self._cache.dump((s_x,s_y,self.name),'sampled_metadata.pik')
        return (s_x,s_y)
        
