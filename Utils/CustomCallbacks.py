#!/usr/bin/env python3
#-*- coding: utf-8

# Training callbacks
from keras.callbacks import Callback

class SaveLRCallback(Callback):
    """
    Get's the last learining rate value after model training is done and save's 
    it for future training.
    """
    def __init__(self,file_name):
        if isinstance(file_name,str):
            self.file_name = file_name

    def on_train_end(self,logs=None):
        """
        Obtains model's last LR and saves it to file
        """
        lr = self.model.optimizer.lr

        try:
            fd = open(self.file_name,'w')
            fd.write("{0:.10f}\n".format(K.eval(lr)))
            fd.close()
        except IOError as e:
            print("[SaveLRCallback] {0}".format(str(e)))

class CalculateF1Score(Callback):
    """
    Calculates F1 score as a callback function. The right way to do it.
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    """

    def on_epoch_end(self,epoch, logs={}):
        y_true = self.model.validation_data[1]

    def on_epoch_begin(self,epoch, logs={}):
        y_true = self.validation_data[1]
