#!/usr/bin/env python3
#-*- coding: utf-8

import os
import argparse
import sys

from multiprocessing import Process

#Project imports
from Preprocessing import Preprocess
from Utils import Exitcodes,CacheManager

#Supported image types
img_types = ['svs', 'dicom', 'nii','tif','tiff', 'png']

def main_exec(config):
    """
    Main execution line. Dispatch processes according to parameter groups.
    Multiple processes here prevent main process from consuming too much memory.
    """

    if config.preprocess:
        proc = Process(target=Preprocess.preprocess_data, args=(config,img_types))
        proc.start()
        proc.join()

        if proc.exitcode != Exitcodes.ALL_GOOD:
            print("System did not end well. Check logs or enhace verbosity level.")
            sys.exit(proc.exitcode)
        
    if config.train:
        pass
    
    if config.postproc:
        pass

    if config.runtest:
        pass

    if not (config.preprocess or config.train or config.postprocess):
        print("The problem begins with choice: preprocess, train or postprocess")

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
        help='Input image or directory of images (runs recursively)',required=True)
    pre_args.add_argument('-predst', dest='predst', type=str,default='tiles', 
        help='Output tiles to directory')
    pre_args.add_argument('-img_type', dest='img_type', type=str, 
        help='Input image type: svs, \
        dicom, nii (Default: \'svs\').',
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
    train_args.add_argument('-b', dest='batch_size', type=int, 
        help='Batch size (Default: 8).', default=8)
    train_args.add_argument('-e', dest='epochs', type=int, 
        help='Number of epochs (Default: 1).', default=1)
    train_args.add_argument('-tn', action='store_true', dest='new_net',
        help='Erase older weights file.')
    train_args.add_argument('-wpath', dest='weights_path',
        help='Use weights file contained in path - usefull for sequential training (Default: None).',
        default=None)
    
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
    parser.add_argument('-v', action='count', default=0, dest='verbose',
        help='Amount of verbosity (more \'v\'s means more verbose).')
    parser.add_argument('-i', action='store_true', dest='info', default=False, 
        help='Return general info about data input, the CNN, etc.')
    parser.add_argument('-logdir', dest='logdir', type=str,default='logs', 
        help='Keep logs of current execution instance in dir.')
    parser.add_argument('-mp', action='store_true', dest='multiprocess', default=False, 
        help='Preprocess multiple images at a time (memory consuming).')
    parser.add_argument('-pb', action='store_true', dest='progressbar', default=False, 
        help='Print progress bars of processing execution.')
    parser.add_argument('-tt', action='store_true', dest='runtest', default=False, 
        help='For debuging only. Run unit tests.')

    config, unparsed = parser.parse_known_args()
    
    #Setup CacheManager - TODO: fill actual files
    files = {
        'tcga.pik':os.path.join(config.presrc,'piks','tcga.pik'),
        'datatree.pik':os.path.join(config.presrc,'datatree.pik'),
        'tiles.pik':os.path.join(config.predst,'tiles.pik')}

    cache_m = CacheManager.CacheManager(locations=files)    

    #Run main program
    main_exec(config)
