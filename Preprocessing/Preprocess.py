#!/usr/bin/env python3
#-*- coding: utf-8

import sys
import os
import numpy as np
import multiprocessing
import tqdm
import concurrent.futures
from skimage import io

from WSIParse import TCGAMerger,GenericData
from Utils import Exitcodes
from Utils import CacheManager

from .ReinhardNormalizer import ReinhardNormalizer

def preprocess_data(config,img_types):
    """
    Main function in preprocessing.

    Works through estipulated configuration.
    """

    #Check SRC and DST directories
    if not os.path.exists(config.presrc):
        if config.verbose > 0:
            print("[Preprocess] No such directory: {0}".format(config.presrc))
        sys.exit(Exitcodes.PATH_ERROR)
    if not os.path.exists(config.predst):
        os.makedirs(config.predst)

    #If SRC dir has already been scanned, no need to redo:
    cache_m = CacheManager.CacheManager(verbose=config.verbose)
    datatree = None
    if config.tcga:
        datatree = cache_m.load('tcga.pik')
        if datatree is None:
            datatree = TCGAMerger.Merger(config.presrc,config.verbose)
            cache_m.dump(datatree,'tcga.pik')
    else:
        datatree = cache_m.load('datatree.pik')
        if datatree is None:
            datatree = GenericData.GenericData(config.presrc,img_types)
            cache_dump(datatree,'datatree.pik')

    #Produce tiles from input images
    #TODO: implement parallel tiling, choose between multiprocess tiling (multiple images processed in parallel) or single process (one image
    #at a time, but work divided in threads
    if config.tile:
        if config.multiprocess:
            multiprocess_run(make_singleprocesstiling,(config,),datatree,step_size=20)
            #make_multiprocesstiling(datatree,config)
        else:
            make_singleprocesstiling(datatree,config)


def make_multiprocesstiling(data,config):
    """
    Generates tiles from input images using multiple processes (process pool).
    """
    step_size = 20
    exec_function = thread_pool_tiler
    dp_path,config.tdim,config.progressbar,config.verbose,process_counter+1

def make_singleprocesstiling(data,config):
    """
    Generates tiles from one input image at a time, but in a multithreaded setup.
    """
    normalizer = ReinhardNormalizer(config.normalize)
    
    for img in data:
        tiles_dir = os.path.join(config.predst,img.getImgName())
        if not os.path.isdir(tiles_dir):
            os.makedirs(tiles_dir)
        thread_pool_tiler(img,config.tdim,config.progressbar,normalizer,config.verbose)


def thread_pool_tiler(img,tsize,progress_bar,normalizer,verbose):
    """
    Creates a thread pool to make tiles of the given image

    @param img <SegImage>: Any class that implements SegImage's methods
    @param tsize <tuple>: (width,height)
    @param progress_bar <bool>: display progress bars
    @param normalizer <str>: Reinhard normalizer instance 
    """
    img_size = img.getImgDim()
    width = img_size[0]
    height = img_size[1]

    max_workers = (((width*height) // (tsize[0]*tsize[1]))/2)
    max_workers = max_workers if max_workers > 1 else 2
    
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    pool_result = []
    counter = 0
    futures = {}
    tile_coords = []

    #Tiles (x,y) top left corner points -
    #TODO: some tiles will need padding later
    margin = 10
    for x in range(1, width, tsize[0]):
        for y in range(1, height, tsize[1]):
            if x + tsize[0] > width - margin:
                pw_x = width - x - margin
            else:
                pw_x = pw
            if y + tsize[1] > height - margin:
                pw_y = height - y - margin
            else:
                pw_y = pw

            if pw_x <= 0 or pw_y <= 0:
                continue
            tile_coords.append((x,y,pw_x,pw_y))
            
    for i in range(len(tile_coords)):
        futures[executor.submit(save_normalize_tile,img,tile_coords[i],normalizer,verbose)] = i

    if progress_bar:
        l = tqdm(desc="Extracting tile...",total=len(tile_coords),position=position)
        
    #for future in concurrent.futures.as_completed(futures):
    for future in futures:
        pool_result.append(future.result())
        if progress_bar:
            l.update(1)
            
    if progress_bar:
        l.close()

    return pool_result


def save_normalize_tile(img,dimensions,normalizer,outdir,verbose):
    """
    Check if tile is background or not, normalize if needed and save to file.

    @param img <SegImage>: any object that implements SegImage
    @param dimensions: tuple (x,y,dx,dy) -> (x,y) point; dw,dy: width,height
    @param outdir <str>: path to output dir (save tiles here)
    @param verbose <int>: verbosity level
    """
    x,y,dx,dy = dimensions
    tile = img.readImageRegion(x,y,dx,dy)

    #Discard background tiles
    if background(tile):
        return None

    #Normalize if whiteness proportion is below 25%:
    if white_ratio(tile) < 0.25:
        tile = normalizer.normalize(tile)

    #TODO: CHECK TILE SIZES AND PAD IF NECESSARY!
    
    #Save tile to disk
    io.imsave(os.path.join(outdir,img.getImgName(),"{0}-{1}_{2}x{3}.png".format(x,y,dx,dy)),tile)
    
    return dimensions

#From https://github.com/SBU-BMI/quip_cnn_segmentation
def white_ratio(pat):
    """
    Foreground/background ratio according to article:
    Spatial Organization And Molecular Correlation Of Tumor-Infiltrating Lymphocytes Using Deep Learning On Pathology Images

    @param pat <np.array>: tile as a numpy array.
    """
    white_count = 0.0
    total_count = 0.001

    if pat.shape[0] < 200 or pat.shape[1] < 200:
        whiteness = background(pat)
        if whiteness:
            return 1.0
        else:
            return 0.0
        
    for x in range(0, pat.shape[0]-200, 100):
        for y in range(0, pat.shape[1]-200, 100):
            p = pat[x:x+200, y:y+200, :]
            whiteness = background(p)
            if whiteness:
                white_count += 1.0
            total_count += 1.0
    return white_count/total_count

def background(pat):
    """
    Foreground/background detection according to article:
    Spatial Organization And Molecular Correlation Of Tumor-Infiltrating Lymphocytes Using Deep Learning On Pathology Images

    @param pat <np.array>: tile as a numpy array.
    """
    whiteness = (np.std(pat[:,:,0]) + np.std(pat[:,:,1]) + np.std(pat[:,:,2])) / 3.0
    if whiteness < 18:
        return True
    else:
        return False
