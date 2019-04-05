#!/usr/bin/env python3
#-*- coding: utf-8

from abc import ABC,abstractmethod
from tqdm import tqdm

import concurrent.futures
import numpy as np
import os

class GenericDS(ABC):
    """
    Generic class for data feeders used to provide training points to Neural Nets.
    """
    def __init__(self,data_path,keepImg=False,verbose=0,pbar=False):
        self.path = None
        if isinstance(data_path,str) and os.path.isdir(data_path):
            self.path = data_path
        else:
            raise ValueError("[GenericImage] Path does not correspond to a file.")

        self._verbose = verbose
        self._pbar = pbar
        self.X = None
        self.Y = None
        
        self._keep = bool(keepImg)


    @abstractmethod
    def _load_metadata_from_dir(self,d):
        pass


    def load_metadata(self):
        """
        Iterates over data patches and creates an instance of a GenericImage subclass for each one
        Returns two tuples of lists (X,Y): X instances of GenericImage subclasses, Y labels;
        """
        files = os.listdir(self.path)

        X,Y = ([],[])
        
        for f in files:
            if os.path.isdir(os.path.join(self.path,f)):
                t_x,t_y = self._load_metadata_from_dir(os.path.join(self.path,f))
                if self._verbose > 1:
                    print(t_x,t_y)
                    
                X.extend(t_x)
                Y.extend(t_y)

        self.X = X
        self.Y = Y
        return X,Y
    
    def load_data(self,split=None,keepImg=False):
        """
        Actually reads images and returns data ready for training
        Returns two tuples of NP arrays (X,Y): X data points, Y labels;

        @param split <tuple>: items are spliting fractions

        If a spliting ratio is provided, return a list of tuples of at most size 3:
        1 - Train;
        2 - Validation;
        3 - Test;
        
        @param keepImg <bool>: Keep image data in memory
        """

        samples = len(self.X)
        y = np.array(self.Y, dtype=np.int32)
        X_data = np.zeros(shape=(tuple([samples] + list(self.X[0].getImgDim()))), dtype=np.int32)
        
        counter = 0
        futures = []

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        for i in range(samples):
            futures.append(executor.submit(self.X[i].readImage,keepImg))

        if self._pbar:
            l = tqdm(desc="Reading images...",total=samples,position=0)
        
        #for future in concurrent.futures.as_completed(futures):
        for i in range(samples):
            X_data[i] = futures[i].result()
            if self._pbar:
                l.update(1)
            
        if self._pbar:
            l.close()

        if split is None:
            return (X_data,y)
        elif sum(split) == 1.0:
            it_count = 0
            split_data = []
            start_idx = 0
            for frac in split:
                it_count = int(frac * samples)
                split_data.append((X_data[start_idx:start_idx+it_count],y[start_idx:start_idx+it_count]))
                start_idx += it_count
            return split_data
                
        else:
            raise ValueError("[GenericDatasource] Spliting values have to equal 1.0")

