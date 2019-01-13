#!/usr/bin/env python3
#-*- coding: utf-8

import sys
import os
import numpy as np
import multiprocessing
import tqdm
import concurrent.futures

from WSIParse import TCGAMerger,GenericData
from Utils import Exitcodes
from Utils import CacheManager

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
            make_multiprocesstiling(datatree,config)
        else:
            make_singleprocesstiling(datatree,config)

    #Normalize tiles
    #TODO: normalize tiles in parallel
    if config.normalize
        pass

def make_multiprocesstiling(data,config):
    """
    Generates tiles from input images using multiple processes (process pool).
    """
    # Perform extractions of frames in parallel and in steps
    step_size = 20
    step = int(len(data) / step_size) + (len(data)%step_size>0)
    datapoints_db = []
    semaphores = []

    process_counter = 0
    pool = multiprocessing.Pool(processes=config.cpu_count,maxtasksperchild=50,
                                    initializer=tqdm.set_lock, initargs=(multiprocessing.RLock(),))

    if config.progressbar:
        l = tqdm(desc="Making tiles...",total=step,position=0)
   
    datapoints = np.asarray(data)
    for i in range(step):
        # get a subset of datapoints
        #end_idx = (i+1)*step_size
        end_idx = step_size
        
        if end_idx > len(data):
            end_idx = len(data)
        
        #cur_datapoints = datapoints[i*step_size : end_idx]
        cur_datapoints = datapoints[:end_idx]

        if config.progressbar:
            semaphores.append(pool.apply_async(thread_pool_tiler,
                                args=(cur_datapoints,dp_path,config.tdim,config.progressbar,config.verbose,process_counter+1),
                                callback=lambda x: l.update(1)))
        else:
            semaphores.append(pool.apply_async(thread_pool_tiler,
                                args=(cur_datapoints,dp_path,config.tdim,config.progressbar,config.verbose,process_counter+1)))
        
        datapoints = np.delete(datapoints,np.s_[:end_idx],axis=0)

        if config.progressbar:
            if process_counter == processes:
                semaphores[process_counter].wait()
                process_counter = 0
            else:
                process_counter += 1

        #datapoints = np.delete(datapoints,np.s_[i*step_size : end_idx],axis=0)        
        #del cur_datapoints    
            
    for i in range(len(semaphores)):
        datapoints_db.extend(semaphores[i].get())
        if not config.progressbar and config.verbose > 0:
            print("[{2}] Done transformations (step {0}/{1})".format(i,len(semaphores)-1,label))

    if config.progressbar:
        l.close()
        print("\n\n\n")

    #Free all possible memory
    pool.close()
    pool.join()

    del datapoints
    
    # remove None points
    return list(filter(lambda x: not x is None, datapoints_db))


def make_singleprocesstiling(data,config):
    """
    Generates tiles from one input image at a time, but in a multithreaded setup.
    """
    for img in data:
        thread_pool_tiler(img,config.tdim,config.progressbar,config.verbose)


def thread_pool_tiler(img,tsize,progress_bar,verbose):
    """
    Creates a thread pool to make tiles of the given image

    @param img <SegImage>: Any class that implements SegImage's methods
    @param tsize <tuple>: (width,height)
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
        futures[executor.submit(save_normalize_tile,img,subset[i],verbose)] = i

    if progress_bar:
        l = tqdm(desc="Extracting tile...",total=len(subset),position=position)
        
    #for future in concurrent.futures.as_completed(futures):
    for future in futures:
        pool_result.append(future.result())
        if progress_bar:
            l.update(1)
            
    if progress_bar:
        l.close()

    return pool_result


#From https://github.com/SBU-BMI/quip_cnn_segmentation
def white_ratio(pat):
    white_count = 0.0
    total_count = 0.001
    for x in range(0, pat.shape[0]-200, 100):
        for y in range(0, pat.shape[1]-200, 100):
            p = pat[x:x+200, y:y+200, :]
            whiteness = (np.std(p[:,:,0]) + np.std(p[:,:,1]) + np.std(p[:,:,2])) / 3.0
            if whiteness < 14:
                white_count += 1.0
            total_count += 1.0
    return white_count/total_count


def save_normalize_tile(img,dimensions,verbose):
    pass
