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
    def __init__(self,data_path,keepImg=False,config=None):
        self.path = None
        if isinstance(data_path,str) and os.path.isdir(data_path):
            self.path = data_path
        else:
            raise ValueError("[GenericImage] Path does not correspond to a file ({0}).".format(data_path))

        self.X = None
        self.Y = None
        self._cache = CacheManager()
        self._keep = keepImg
        self._cpu_count = config.cpu_count if not config is None else 1
        self._verbose = config.verbose if not config is None else 0
        self._pbar = config.progressbar if not config is None else False
        self._config = config


    @abstractmethod
    def _load_metadata_from_dir(self,d):
        pass

    @abstractmethod
    def get_dataset_dimensions(self):
        pass

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

    def split_metadata(self,split):
        """
        Returns all metadata split into N sets, defined by the spliting tuples
        """
        return self._split_data(split,self.X,self.Y)
    
    def load_metadata(self):
        """
        Iterates over data patches and creates an instance of a GenericImage subclass for each one
        Returns two tuples of lists (X,Y): X instances of GenericImage subclasses, Y labels;

        OBS: Dataset metadata is shuffled once here. Random sample generation is done during training.
        """
        files = os.listdir(self.path)

        X,Y = ([],[])
        
        if self._cache.checkFileExistence('metadata.pik'):
            X,Y = self._cache.load('metadata.pik')
            if self._verbose > 0:
                print("[GenericDatasource] Loaded split data cache. Used previously defined splitting.")
        else:
            dlist = []
            for f in files:
                item = os.path.join(self.path,f)
                if os.path.isdir(item):
                    dlist.append(item)

            mdata = multiprocess_run(self._run_multiprocess,tuple(),dlist,
                                         self._cpu_count,self._pbar,
                                         step_size=1,output_dim=2,txt_label='directories',verbose=self._verbose)

            
            X.extend(mdata[0]) #samples
            Y.extend(mdata[1]) #labels
                    
            #Shuffle samples and labels maintaining relative order
            combined = list(zip(X,Y))
            random.shuffle(combined)
            X[:],Y[:] = zip(*combined)
            
            self._cache.dump((X,Y),'metadata.pik')
            
        self.X = X
        self.Y = Y
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
            
        samples = len(X)
        y = np.array(Y, dtype=np.int32)
        dataset_dim = self.get_dataset_dimensions()[0]
        img_dim = dataset_dim[1:]
        X_data = np.zeros(shape=(samples,)+img_dim, dtype=np.float32)
        
        counter = 0
        futures = []

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=7)
        for i in range(samples):
            futures.append(executor.submit(X[i].readImage,keepImg,img_dim))

        if self._pbar:
            l = tqdm(desc="Reading images...",total=samples,position=0)
        
        #for future in concurrent.futures.as_completed(futures):
        for i in range(samples):
            X_data[i] = futures[i].result()
            if self._pbar:
                l.update(1)
            elif self._verbose > 0:
                print("[load_data] Read image: {0}".format(i))
            
        if self._pbar:
            l.close()

        if split is None:
            return (X_data,y)
        else:
            return self._split_data(split,X_data,y)
