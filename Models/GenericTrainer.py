#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
import os,sys
import re
import numpy as np

from Datasources.CellRep import CellRep
from Utils import SaveLRCallback,CalculateF1Score
from Utils import Exitcodes,CacheManager

#Keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
# Training callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,LearningRateScheduler
from keras.utils import to_categorical

import tensorflow as tf

def run_training(config,locations=None):
    """
    Main training function, to work as a new process
    """
    if config.info:
        print("Starting training process....")

    if not locations is None:
        cache_m = CacheManager(locations=locations)
    trainer = Trainer(config)
    trainer.run()

def _reduce_lr_on_epoch(epoch,lr):
    #Reduces LR by a factor of 10 every 12 epochs
    if epoch > 0 and not (epoch%12):
        lr /= 10
    return lr

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
        @param config <parsed configurations>: configurations
        """
        self._config = config
        self._verbose = config.verbose
        self._ds = None
        self._rex = None

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
            self._ds = getattr(dsm,self._config.data)(self._config.predst,self._config.keepimg,self._config)
        else:
            self._ds = CellRep(self._config.predst,self._config.keepimg,self._config)

        net_module = importlib.import_module('Models',net_name)
        net_model = getattr(net_module,net_name)(self._config,self._ds)

        self._ds.load_metadata()

        self._rex = r'{0}-t(?P<try>[0-9]+)e(?P<epoch>[0-9]+).h5'.format(net_model.name)
        
        if self._config.delay_load:
            return self.train_model_iterator(net_model)
        else:
            return self.train_model(net_model)
    
    def train_model(self,model):
        """
        Execute training according to configurations. Load all data at once.

        @param model <Keras trainable model>: model to be trained
        """
        rcomp = re.compile(self._rex)
        
        train,val,_ = self._ds.load_data(split=self._config.split,keepImg=False)
                    
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


        single,parallel = model.build()
        if not parallel is None:
            training_model = parallel
        else:
            training_model = single
        
        # try to resume the training
        weights = list(filter(lambda f: f.endswith(".h5") and f.startswith(model.name),os.listdir(self._config.weights_path)))
        weights.sort()
        old_e_offset = 0
        if len(weights) > 0 and not self._config.new_net:
            # get last file (which is the furthest on the training) if exists
            ep_weights_file = weights[len(weights)-2]
            match = rcomp.fullmatch(ep_weights_file)
            if match:
                old_e_offset = int(match.group('epoch'))
            else:
                old_e_offset = 0

            # load weights
            try:
                single.load_weights(os.path.join(self._config.weights_path,
                    ep_weights_file))
                if self._verbose > 0:
                    print("Sucessfully loaded previous weights: {0}".format(ep_weights_file))
            except ValueError:
                single.load_weights(os.path.join(self._config.weights_path,"{0}-weights.h5".format(model.name)))
                if self._verbose > 0:
                    print("Sucessfully loaded previous weights from consolidated file.")
            except ValueError:
                print("[ALERT] Could not load previous weights, training from scratch")
                
        wf_header = "{0}-t{1}".format(model.name,old_e_offset+1)

        ### Define special behaviour CALLBACKS
        callbacks = []
        ## ModelCheckpoint
        callbacks.append(ModelCheckpoint(os.path.join(
            self._config.weights_path, wf_header + "e{epoch:02d}.h5"), 
            save_weights_only=True, period=1,save_best_only=True,monitor='val_acc'))
        ## ReduceLROnPlateau
        callbacks.append(ReduceLROnPlateau(monitor='val_loss',factor=0.4,\
                                           patience=3,verbose=self._verbose,\
                                           mode='auto',min_lr=1e-6))        

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
            callbacks=callbacks)

        #Weights should be saved only through the plain model
        cache_m = CacheManager()
        single.save_weights(model.get_weights_cache())
        single.save(model.get_model_cache())
        cache_m.dump(self._config.split,'split_ratio.pik')

        return Exitcodes.ALL_GOOD
    
    def train_model_iterator(self,model):
        """
        Use the fit_iterator to control the sample production
        TODO: for future use
        """
        from Models import ThreadedGenerator

        rcomp = re.compile(self._rex)
        
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

        #Setup of generators, augmentation, preprocessing
        train_data,val_data,_ = self._ds.split_metadata(self._config.split)
        
        if self._verbose > 0:
            unique,count = np.unique(train_data[1],return_counts=True)
            l_count = dict(zip(unique,count))
            print("Train labels: {0} are 0; {1} are 1;\n - {2:.2f} are positives".format(l_count[0],l_count[1],(l_count[1]/(l_count[0]+l_count[1]))))
            
            unique,count = np.unique(val_data[1],return_counts=True)
            l_count = dict(zip(unique,count))
            print("Validation labels: {0} are 0; {1} are 1;\n - {2:.2f} are positives".format(l_count[0],l_count[1],(l_count[1]/(l_count[0]+l_count[1]))))
            
        if self._config.info:
            print("Train set: {0} items".format(len(train_data[0])))
            print("Validate set: {0} items".format(len(val_data[0])))
            
        train_prep = ImageDataGenerator(
            samplewise_center=self._config.batch_norm,
            samplewise_std_normalization=self._config.batch_norm,
            rotation_range=180,
            width_shift_range=20,
            height_shift_range=20,
            zoom_range=.2,
            #shear_range=.05,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=(-20.0,20.0))

        if not self._config.tile is None:
            fix_dim = self._config.tile
        else:
            fix_dim = self._ds.get_dataset_dimensions()[0][1:] #Only smallest image dimensions matter here
        train_generator = ThreadedGenerator(dps=train_data,
                                            classes=self._ds.nclasses,
                                            dim=fix_dim,
                                            batch_size=self._config.batch_size,
                                            image_generator=train_prep,
                                            shuffle=True,
                                            verbose=self._config.verbose)

        val_prep = ImageDataGenerator(
            samplewise_center=self._config.batch_norm,
            samplewise_std_normalization=self._config.batch_norm)
        val_generator = ThreadedGenerator(dps=val_data,
                                            classes=self._ds.nclasses,
                                            dim=fix_dim,
                                            batch_size=self._config.batch_size,
                                            image_generator=val_prep,
                                            shuffle=True,
                                            verbose=self._config.verbose)

        single,parallel = model.build()
        if not parallel is None:
            training_model = parallel
        else:
            training_model = single
            
        # try to resume the training
        weights = list(filter(lambda f: f.endswith(".h5") and f.startswith(model.name),os.listdir(self._config.weights_path)))
        weights.sort()
        old_e_offset = 0
        if len(weights) > 0 and not self._config.new_net:
            # get last file (which is the furthest on the training) if exists
            ep_weights_file = weights[len(weights)-2]
            match = rcomp.fullmatch(ep_weights_file)
            if match:
                old_e_offset = int(match.group('epoch'))
            else:
                old_e_offset = 0
            # load weights
            try:
                single.load_weights(os.path.join(self._config.weights_path,
                    ep_weights_file))
                if self._verbose > 0:
                    print("Sucessfully loaded previous weights: {0}".format(ep_weights_file))
            except ValueError:
                single.load_weights(os.path.join(self._config.weights_path,"{0}-weights.h5".format(model.name)))
                if self._verbose > 0:
                    print("Sucessfully loaded previous weights from consolidated file.")
            except ValueError:
                print("[ALERT] Could not load previous weights, training from scratch")
                
        wf_header = "{0}-t{1}".format(model.name,old_e_offset+1)

        ### Define special behaviour CALLBACKS
        callbacks = []
        ## ModelCheckpoint
        callbacks.append(ModelCheckpoint(os.path.join(
            self._config.weights_path, wf_header + "e{epoch:02d}.h5"), 
            save_weights_only=True, period=1,save_best_only=True,monitor='val_acc'))
        ## ReduceLROnPlateau
        callbacks.append(ReduceLROnPlateau(monitor='loss',factor=0.5,\
                                           patience=3,verbose=self._verbose,\
                                           mode='auto',min_lr=1e-8))
        callbacks.append(LearningRateScheduler(_reduce_lr_on_epoch,verbose=1))
        #callbacks.append(CalculateF1Score())

        if self._config.info:
            print(single.summary())

        training_model.fit_generator(
            generator = train_generator,
            steps_per_epoch = len(train_data[0]) // self._config.batch_size,
            epochs = self._config.epochs,
            validation_data = val_generator,
            validation_steps = len(val_data[0]) //self._config.batch_size,
            verbose = 1 if self._verbose > 0 else 0,
            use_multiprocessing = False,
            workers=self._config.cpu_count*2,
            max_queue_size=self._config.batch_size*3,
            callbacks=callbacks,
            )

        #Weights should be saved only through the plain model
        cache_m = CacheManager()
        single.save_weights(model.get_weights_cache())
        single.save(model.get_model_cache())
        cache_m.dump(tuple(self._config.split),'split_ratio.pik')
        
        return Exitcodes.ALL_GOOD
        
