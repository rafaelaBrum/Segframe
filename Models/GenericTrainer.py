#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os,sys

from Datasources.CellRep import CellRep
from Utils import SaveLRCallback
from Utils import Exitcodes,CacheManager

#Keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
# Training callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical

import tensorflow as tf

def run_training(config,locations=None):
    """
    Main training function, to work as a new process
    """
    if config.info:
        print("Starting training process.")

    if not locations is None:
        cache_m = CacheManager(locations=locations)
    trainer = Trainer(config)
    trainer.run()
    
class Trainer(object):
    """
    Class that implements the training procedures applicable to all
    CNN models.

    Specialized training my be needed for some models and those should be 
    implemented elsewhere.

    @param config <argparse config>: configurations as specified by user
    @param ds <datasource>: some subclass of a GenericDatasource
    """

    def __init__(self,config):
        """
        @config <parsed configurations>: configurations
        """
        self._config = config
        self._verbose = config.verbose
        self._ds = None

    def run(self):
        """
        Checks configurations, loads correct module, loads data
        Trains!

        New networks should be inserted as individual modules. Networks should be imported
        by the Models module.
        """
        net_name = self._config.network
        if net_name is None or net_name == '':
            print("A network should be specified")
            return Exitcodes.RUNTIME_ERROR
                
        if self._config.data:
            dsm = importlib.import_module('Datasources',self._config.data)
            self._ds = getattr(dsm,self._config.data)(self._config.predst,self._config.keepimg,self._config.verbose,self._config.progressbar)
        else:
            self._ds = CellRep(self._config.predst,self._config.keepimg,self._config.verbose,self._config.progressbar)

        net_module = importlib.import_module('Models',net_name)
        net_model = getattr(net_module,net_name)(self._config,self._ds)

        self._ds.load_metadata()
        train,val,_ = self._ds.load_data(split=self._config.split,keepImg=False)
                
        return self.train_model(net_model,train,val)
    
    def train_model(self,model,train,val):
        """
        Execute training according to configurations. 

        @param model <Keras trainable model>: model to be trained
        @param train <numpy array>: training data
        @param val <numpy array>: validation data
        """
        x_train,y_train = train
        x_val,y_val = val

        #Labels should be converted to categorical representation
        y_train = to_categorical(y_train,self._ds.nclasses)
        y_val = to_categorical(y_val,self._ds.nclasses)
        
        # session setup
        sess = K.get_session()
        ses_config = tf.ConfigProto(
            device_count={"CPU":self._config.cpu_count,"GPU":self._config.gpu_count},
            intra_op_parallelism_threads=self._config.cpu_count if self._config.gpu_count == 0 else self._config.gpu_count, 
            inter_op_parallelism_threads=self._config.cpu_count if self._config.gpu_count == 0 else self._config.gpu_count,
            log_device_placement=True if self._verbose > 1 else False
            )
        sess.config = ses_config
        K.set_session(sess)

        train_generator = ImageDataGenerator(
            samplewise_center=False,
            samplewise_std_normalization=False,
            rotation_range=10,
            width_shift_range=.1,
            height_shift_range=.1,
            zoom_range=.08,
            shear_range=.03,
            horizontal_flip=True,
            vertical_flip=True)

        val_generator = ImageDataGenerator(
            samplewise_center=False,
            samplewise_std_normalization=False)

        # try to resume the training
        weights = list(filter(lambda f: f.endswith(".hdf5") and f.startswith(model.name),os.listdir(self._config.model_path)))
        weights.sort()
        old_e_offset = 0
        if len(weights) > 0 and not self._config.new_net:
            # get last file (which is the furthest on the training) if exists
            ep_weights_file = weights[len(weights)-1]
            old_e_offset = int(ep_weights_file.split(
                ".hdf5")[0].split('-')[1].split("e")[0].split("t")[1])

            # load weights
            try:
                model.load_weights(os.path.join(self._config.model_path,
                    ep_weights_file))
                if self._verbose > 0:
                    print("Sucessfully loaded previous weights: {0}".format(ep_weights_file))
            except ValueError:
                model.load_weights(os.path.join(self._config.model_path,"{0}_cnn_weights.h5".format(model.name)))
                if self._verbose > 0:
                    print("Sucessfully loaded previous weights from consolidated file.")
            except ValueError:
                print("[ALERT] Could not load previous weights, training from scratch")
                
        wf_header = "{0}-t{1}".format(model.name,old_e_offset+1)

        ### Define special behaviour CALLBACKS
        callbacks = []
        ## ModelCheckpoint
        callbacks.append(ModelCheckpoint(os.path.join(
            self._config.weights_path, wf_header + "e{epoch:02d}.hdf5"), 
            save_weights_only=True, period=1,save_best_only=True,monitor='val_acc'))
        ## ReduceLROnPlateau
        callbacks.append(ReduceLROnPlateau(monitor='val_loss',factor=0.4,\
                                           patience=3,verbose=self._verbose,\
                                           mode='auto',min_lr=1e-6))        

        single,parallel = model.build()
        if not parallel is None:
            training_model = parallel
        else:
            training_model = single

        if self._config.info:
            print(training_model.summary())

        training_model.fit_generator(
            generator = train_generator.flow(x_train,y_train,batch_size=self._config.batch_size,shuffle=True),
            steps_per_epoch = len(x_train) // self._config.batch_size,
            epochs = self._config.epochs,
            validation_data = val_generator.flow(x_val,y_val,batch_size=1),
            validation_steps = len(x_val) //self._config.batch_size,
            verbose = 1 if self._verbose > 0 else 0,
            use_multiprocessing = False,
            workers=3,
            max_queue_size=45,
            callbacks=callbacks,
            )

        #Weights should be saved only through the plain model
        single.save_weights(model.get_weights_cache())
        single.save(cache_m.fileLocation(model.get_model_cache()))

        return Exitcodes.ALL_GOOD
    
    def train_model_iterator(self,model,train_it,val_it):
        """
        Use the fit_iterator to control the sample production
        TODO: for future use
        """
        pass
