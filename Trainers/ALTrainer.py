#!/usr/bin/env python3
#-*- coding: utf-8

import os,sys
import numpy as np

from GenericTrainer import Trainer

def run_training(config,locations=None):
    """
    Main training function, to work as a new process
    """
    if config.info:
        print("Starting training process....")

    if not locations is None:
        cache_m = CacheManager(locations=locations)
    trainer = ALTrainer(config)
    trainer.run()
    
class ALTrainer(Trainer):
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
