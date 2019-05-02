#!/usr/bin/env python3
#-*- coding: utf-8

import os
import argparse
import sys

import multiprocessing as mp
from multiprocessing import Process

#Project imports
from Preprocessing import Preprocess
from Utils import Exitcodes,CacheManager
from Testing import TrainTest,DatasourcesTest
from Models import GenericTrainer

#Supported image types
img_types = ['svs', 'dicom', 'nii','tif','tiff', 'png']

def main_exec(config):
    """
    Main execution line. Dispatch processes according to parameter groups.
    Multiple processes here prevent main process from consuming too much memory.
    """

    if not os.path.isdir(config.bdir):
        os.mkdir(config.bdir)

    if config.preprocess:
        if config.multiprocess:
            proc = Process(target=Preprocess.preprocess_data, args=(config,img_types))
            proc.start()
            proc.join()

            if proc.exitcode != Exitcodes.ALL_GOOD:
                print("System did not end well. Check logs or enhace verbosity level.")
                sys.exit(proc.exitcode)
        else:
            Preprocess.preprocess_data(config,img_types)
        
    if config.train:
        if config.multiprocess:
            ctx = mp.get_context('spawn')
            cache_m = CacheManager()
            proc = ctx.Process(target=GenericTrainer.run_training, args=(config,cache_m.getLocations()))
            proc.start()
            proc.join()

            if proc.exitcode != Exitcodes.ALL_GOOD:
                print("System did not end well. Check logs or enhace verbosity level.")
                sys.exit(proc.exitcode)
        else:
            GenericTrainer.run_training(config,None)
    
    if config.postproc:
        pass

    if config.runtest:
        if config.tmode == 0:
            pass
        elif config.tmode == 1:
            #Run train test
            TrainTest.run(config)
        elif config.tmode == 2:
            DatasourcesTest.run(config)

    if not (config.preprocess or config.train or config.postproc or config.runtest):
        print("The problem begins with choice: preprocess, train, postprocess or test")

if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Convolunional Neural \
        Network for Image Segmentation.')

    ##Preprocessing options
    pre_args = parser.add_argument_group('Preprocessing','Includes image format usage, tiling and normalization')
    arg_groups.append(pre_args)
    
    pre_args.add_argument('--pre', action='store_true', dest='preprocess', default=False, 
        help='Run preprocess steps')
    pre_args.add_argument('-tile', action='store_true', dest='tile', default=False, 
        help='Make tiles from input images')
    pre_args.add_argument('-tcga', action='store_true', dest='tcga', default=False, 
        help='Input is a TCGA image base.')    
    pre_args.add_argument('-presrc', dest='presrc', type=str,default='', 
        help='Input image or directory of images (runs recursively)',required=False)
    pre_args.add_argument('-predst', dest='predst', type=str,default='tiles', 
        help='Output tiles to directory')
    pre_args.add_argument('-img_type', dest='img_type', type=str, 
        help='Input image type: svs, dicom, nii (Default: \'svs\').',
        choices=img_types, default='svs')
    pre_args.add_argument('-mag', dest='magnification', type=int, 
        help='For SVS images only, use specific magnification level.',
        choices=[2,4,8,10,20,40],default=40)
    pre_args.add_argument('-tdim', dest='tile', nargs=2, type=int, 
        help='Tile width and heigth (Default: 200 200 for SVS 50 um).', 
        default=(200, 200), metavar=('Width', 'Height'))
    pre_args.add_argument('-norm', dest='normalize', type=str,default='Preprocessing/target_40X.png', 
        help='Normalize tiles based on reference image (given)')
    

    ##Training options
    train_args = parser.add_argument_group('Training','Common network training options')
    arg_groups.append(train_args)

    train_args.add_argument('--train', action='store_true', dest='train', default=False, 
        help='Train model')
    train_args.add_argument('-net',dest='network',type=str,default='',help='Network name which should be trained.\n \
    Check documentation for available models.')
    train_args.add_argument('-data',dest='data',type=str,help='Dataset name to train model.\n \
    Check documentation for available datasets.',default='')
    train_args.add_argument('-b', dest='batch_size', type=int, 
        help='Batch size (Default: 8).', default=8)
    train_args.add_argument('-e', dest='epochs', type=int, 
        help='Number of epochs (Default: 1).', default=1)
    train_args.add_argument('-tn', action='store_true', dest='new_net',
        help='Do not use older weights file.')
    train_args.add_argument('-wpath', dest='weights_path',
        help='Use weights file contained in path - usefull for sequential training (Default: None).',
        default='ModelWeights')
    train_args.add_argument('-split', dest='split', nargs=3, type=float, 
        help='Split data in as much as 3 sets (Default: 80%% train, 10%% validation, 10%% test).',
        default=(0.8, 0.1,0.1), metavar=('Train', 'Validation','Test'))
    
    ##Postprocessing options
    post_args = parser.add_argument_group('Postprocessing', 'Generate bounding boxes or other operation')
    arg_groups.append(post_args)

    post_args.add_argument('--post', action='store_true', dest='postproc', default=False, 
        help='Run postprocess steps')
    post_args.add_argument('-postsrc', dest='postsrc', type=str,default='tiles', 
        help='Input image or directory of images (runs recursively)')
    post_args.add_argument('-postdst', dest='postdst', type=str,default='', 
        help='Output tiles to directory. If empty, output to same directory as input')
    
    ##Model selection
    model_args = parser.add_argument_group('Model')
    arg_groups.append(model_args)

    model_args.add_argument('-model_dir', dest='model_path',
        help='Save trained models in dir (Default: TrainedModels).',
        default='TrainedModels')
    
    ##Hardware configurations
    hd_args = parser.add_argument_group('Hardware')
    arg_groups.append(hd_args)

    hd_args.add_argument('-gpu', dest='gpu_count', type=int, 
        help='Number of GPUs available (Default: 0).', default=0)
    hd_args.add_argument('-cpu', dest='cpu_count', type=int, 
        help='Number of CPU cores available (Default: 1).', default=1)

    ##Runtime options
    pre_args.add_argument('-out', dest='bdir', type=str,default='', 
        help='Base dir to store all temporary data and general output',required=True)
    pre_args.add_argument('-cache', dest='cache', type=str,default='cache', 
        help='Keeps caches in this directory',required=False)
    parser.add_argument('-v', action='count', default=0, dest='verbose',
        help='Amount of verbosity (more \'v\'s means more verbose).')
    parser.add_argument('-i', action='store_true', dest='info', default=False, 
        help='Return general info about data input, the CNN, etc.')
    parser.add_argument('-logdir', dest='logdir', type=str,default='logs', 
        help='Keep logs of current execution instance in dir.')
    parser.add_argument('-mp', action='store_true', dest='multiprocess', default=False, 
        help='[TODO] Preprocess multiple images at a time (memory consuming - multiple processes).')
    parser.add_argument('-pb', action='store_true', dest='progressbar', default=False, 
        help='Print progress bars of processing execution.')
    parser.add_argument('-k', action='store_true', dest='keepimg', default=False, 
        help='Keep loaded images in memory.')
    parser.add_argument('-d', action='store_true', dest='delay_load', default=True, 
        help='Delay the loading of images to the latest moment possible (memory efficiency).')

    ##System tests
    test_args = parser.add_argument_group('Tests')
    arg_groups.append(test_args)
    
    parser.add_argument('-t', action='store_true', dest='runtest', default=False, 
        help='Run tests.')
    test_args.add_argument('-tmode', dest='tmode', type=int, 
        help='Run tests for individual subsystems: \n \
        0 - Run all tests; \n \
        1 - Run training test; \n \
        2 - Run Datasources test;',
       choices=[0,1,2],default=0)
        
    config, unparsed = parser.parse_known_args()
    
    #Setup CacheManager - TODO: fill actual files
    files = {
        'tcga.pik':os.path.join(config.presrc,'piks','tcga.pik'),
        'split_data.pik':os.path.join(config.cache,'split_data.pik'),
        'data_dims.pik':os.path.join(config.cache,'data_dims.pik'),
        'tiles.pik':os.path.join(config.predst,'tiles.pik'),
        'cae_model.h5':os.path.join(config.model_path,'cae_model.h5'),
        'vgg16_weights_notop.h5':os.path.join(config.model_path,'vgg16_weights_notop.h5')}

    cache_m = CacheManager(locations=files)    

    #Run main program
    main_exec(config)
