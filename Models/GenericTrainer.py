#!/usr/bin/env python3
#-*- coding: utf-8

import importlib
from Datasources.CellRep import CellRep

#Keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

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

        New networks should be inserted as individual modules with both
        module and network receiving the same name
        """
        net_name = config.network
        net_module = importlib.import_module(net_name,net_name)
        if self._config.data:
            self._ds = importlib.import_module('Datasources',self._config.data)()
        else:
            self._ds = CellRep()

        t_x,t_y = self._ds.load_data(split=self._config.split,keepImg=False)

        model = net_module.build()
                
        self.train_model(model,t_x,t_y)
    
    def train_model(self,model,x_train,y_train,x_val,y_val):
        """
        Execute training according to configurations. 

        @param model <Keras trainable model>: model to be trained
        @param x_train <numpy array>: training data
        @param y_train <numpy array>: training labels
        @param x_val <numpy array>: validation data
        @param y_val <numpy array>: validation labels
        """
        
        # session setup
        sess = K.get_session()
        ses_config = tf.ConfigProto(
            device_count={"CPU":args.cpu_count,"GPU":args.gpu_count},
            intra_op_parallelism_threads=self._config.cpu_count if self._config.gpu_count == 0 else self._config.gpu_count, 
            inter_op_parallelism_threads=self._config.cpu_count if self._config.gpu_count == 0 else self._config.gpu_count,
            log_device_placement=True if self._verbose > 1 else False
            )
        sess.config = ses_config
        K.set_session(sess)

        train_generator = ImageDataGenerator(
            samplewise_center=False,
            samplewise_std_normalization=False)
            #rotation_range=10,
            #width_shift_range=.1,
            #height_shift_range=.1,
            #zoom_range=.08,
            #shear_range=.03,
            #horizontal_flip=True,
            #vertical_flip=True)

        val_generator = ImageDataGenerator(
            samplewise_center=False,
            samplewise_std_normalization=False)

        model.fit_generator(
            generator = train_generator.flow(x_train,y_train,batch_size=self._config.batch_size),
            steps_per_epoch = len(x_train) // self._config.batch_size,
            epochs = self._config.epochs,
            validation_data = val_generator.flow(x_val,y_val,batch_size=1),
            validation_steps = len(x_val) // batch_size,
            verbose = 1 if self._verbose > 0 else 0,
            use_multiprocessing = False,
            workers=3,
            max_queue_size=45,
            callbacks=callbacks,
            )


    def train_model_iterator(self,model,train_it,val_it):
        """
        Use the fit_iterator to control the sample production
        """
        pass
