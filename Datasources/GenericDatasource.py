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
    def __init__(self,data_path,verbose=0,pbar=False):
        self.path = None
        if isinstance(path,str) and os.path.isdir(data_path):
            self.path = data_path
        else:
            raise ValueError("[GenericImage] Path does not correspond to a file.")

        self._verbose = verbose
        self._pbar = pbar
        self.X = None
        self.Y = None        


    @abstractmethod
    def _load_metadata_from_dir(self,d):
        pass


    def load_metadata(self):
        """
        Iterates over data patches and creates an instance of a GenericImage subclass for each one
        Returns a tuple of lists (X,Y): X instances of GenericImage subclasses, Y labels
        """
        files = os.listdir(self.path)

        X,Y = ([],[])
        
        for f in files:
            if os.path.isdir(f):
                t_x,t_y = self._load_metadata_from_dir(os.path.join(self.path,f))
                X.extend(t_x)
                Y.extend(t_y)

        self.X = X
        self.Y = Y
        return X,Y
    
    def load_data(self):
        """
        Actually reads images and returns data ready for training
        Returns a tuple of NP arrays (X,Y): X training points, Y labels
        """

        samples = len(self.X)
        y = np.array(self.Y, dtype=np.int32)
        X_data = np.zeros(shape=(tuple([samples] + list(self.X[0].getImgDim()))), dtype=np.float32)
        
        counter = 0
        futures = []

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        for i in range(samples):
            futures[i] = executor.submit(self.X[i].readImage)

        if self._pbar:
            l = tqdm(desc="Reading images...",total=samples,position=position)
        
        #for future in concurrent.futures.as_completed(futures):
        for i in range(samples):
            X_data[i] = future.result()
            if self._pbar:
                l.update(1)
            
        if self._pbar:
            l.close()

        return (X_data,y)
