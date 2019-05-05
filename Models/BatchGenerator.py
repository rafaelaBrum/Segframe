#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

import keras
from keras.utils.data_utils import Sequence
from keras.preprocessing.image import Iterator,ImageDataGenerator
from keras import backend as K

#System modules
import threading
import numpy as np
import timeit

class GenericIterator(Iterator):
    """
        RHDIterator is actually a generator, yielding the data tuples from a data source as a correlation list.
        
        # Arguments
        image_data_generator: Instance of `ImageDataGenerator`
        data: tuple (X,Y) where X are samples, Y are corresponding labels
        to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
    """

    def __init__(self,
                     data,
                     classes,
                     dim=None,
                     batch_size=8,
                     image_generator=None,
                     shuffle=True,
                     seed=173,
                     black_region=None,
                     data_mean=0.0,
                     verbose=0):

        self.data = data
        self.classes = classes
        self.dim = dim
        self.black_region = black_region
        self.mean = data_mean
        self.image_generator = None
        self.verbose = verbose

        #Keep information of example shape as soon as the information is available
        self.shape = None
        
        if not image_generator is None and isinstance(image_generator,ImageDataGenerator):
            self.image_generator = image_generator
        elif not image_generator is None:
            raise TypeError("Image generator should be an " \
            "ImageDataGenerator instance")

        #TODO: Data length needs adjustment, total data is not necessarily the number of exam directories
        super(GenericIterator, self).__init__(n=len(self.data), batch_size=batch_size, shuffle=shuffle, seed=seed)


    def returnDataSize(self):
        """
        Returns the number of examples
        """
        return len(self.data)
    

    def next(self):
        """
        For python 2.x.
        # Returns
        The next batch.
        """
            
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)

    def returnNumClasses(self):
        return len(self.classes)
            
    def returnDataInOrder(self,idx):
        """
        Returns a data batch, starting in position idx
        """
        index_array = [i for i in range(idx,idx+self.batch_size)]
        # Check which element(s) to use
        return self._get_batches_of_transformed_samples(index_array)

class SingleGenerator(GenericIterator):
    """
    Generates batches of images, applies augmentation, resizing, centering...the whole shebang.
    """
    def __init__(self, 
                     dps,
                     classes,
                     dim=None,
                     batch_size=8,
                     image_generator=None,
                     shuffle=True,
                     seed=173,
                     black_region=None,
                     data_mean=0.0,
                     verbose=0,
                     variable_shape=False):
        
        #Set True if examples in the same dataset can have variable shapes
        self.variable_shape = variable_shape
        
        super(SingleGenerator, self).__init__(data=dps,
                                                classes=classes,
                                                dim=dim,
                                                batch_size=batch_size,
                                                image_generator=image_generator,
                                                shuffle=shuffle,
                                                seed=seed,
                                                black_region=black_region,
                                                data_mean=data_mean,
                                                verbose=verbose)


    def _get_batches_of_transformed_samples(self,index_array):
        """
        Only one argument will be considered. The index array has preference

        #Arguments
           index_array: array of sample indices to include in batch; or
        # Returns 
            a batch of transformed samples
        """
        # calculate dimensions of each data point
        #Should only create the batches of appropriate size
        if not self.shape is None:
            batch_x = np.zeros(tuple([len(index_array)] + list(self.shape)), dtype=K.floatx())
        else:
            batch_x = None
        y = np.zeros(tuple([len(index_array)]),dtype=int)
                
        # initialize output lists
        #batch_x = np.zeros((len(index_array), self.n_frames) + input_shape, 
        #    dtype=K.floatx())
        #y = np.zeros(len(index_array), dtype=int)

        # generate a random batch of points
        X = self.data[0]
        Y = self.data[1]
        for i,j in enumerate(index_array):
            t_x = X[j]
            t_y = Y[j]
                
            example = t_x.readImage(size=self.dim,verbose=self.verbose)
            
            if batch_x is None:
                self.shape = example.shape
                batch_x = np.zeros(tuple([len(index_array)] + list(self.shape)),dtype=K.floatx())
            
            #TEST PURPOSES ONLY - This is slow given the sizes
            #involved
            if not self.image_generator is None:
                example = self.image_generator.random_transform(example,self.seed)
                example = self.image_generator.standardize(example)

            # add point to x_batch and diagnoses to y
            batch_x[i] = example
            y[i] = t_y

        #Center data
        #batch_x -= self.mean
        #Normalize data pixels
        #batch_x /= 255

        if self.variable_shape:
            self.shape = None
            
        output = (batch_x, keras.utils.to_categorical(y, self.classes))
        return output         
