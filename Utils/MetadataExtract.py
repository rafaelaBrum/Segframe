#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

import numpy as np
import os
import shutil
import sys
import argparse
import pickle

def process_al_metadata(config):
    if not os.path.isdir(config.out_dir):
        os.mkdir(config.out_dir)

    if not os.path.isdir(config.sdir):
        print("Directory not found: {}".format(config.sdir))
        sys.exit(1)

    files = os.listdir(config.sdir)

    acfiles = {}
    for f in files:
        if f.startswith('al-metadata'):
            ac_id = int(f.split('.')[0].split('-')[3][1:])
            acfiles[ac_id] = os.path.join(config.sdir,f)

    ordered_k = list(acfiles.keys())
    ordered_k.sort()
    initial_set = None
    ac_imgs = {}
    for k in ordered_k:
        with open(acfiles[k],'rb') as fd:
            train,val,test = pickle.load(fd)
            
        if initial_set is None:
            #Acquisitions are obtained from keys k and k-1
            initial_set = list(train[0])
        else:
            imgs = np.setdiff1d(list(train[0]),initial_set,True)
            print("Acquired {} images in acquisition {}".format(imgs.shape[0],k-1))
            ac_imgs[k-1] = imgs
            initial_set = list(train)
            
    acquisitions = [ac_imgs[k][:config.n] for k in config.ac_n]

    print("# of acquisitions obtained: {} -> ".format(len(acquisitions)),end='')
    print(" ".join([str(len(d)) for d in acquisitions]))
    print("Copying...")

    same_name = {}
    for a in range(len(config.ac_n)):
        ac_path = os.path.join(config.out_dir,str(config.ac_n[a]))
        if not os.path.isdir(ac_path):
            os.mkdir(ac_path)
        for img in acquisitions[a]:
            img_name = os.path.basename(img.getPath())
            if img_name in same_name:
                print("Images with the same name detected: {}\n{}\n{}".format(img_name,img.getPath(),same_name[img_name]))
            else:
                same_name[img_name] = img.getPath()
            shutil.copy(img.getPath(),ac_path)

    print("Acquired images copied to output dir.")

def process_cluster_metadata(config):
    def _copy_img(dst_dir,acq,cln,img,img_in):
        ac_path = os.path.join(dst_dir,str(acq),str(cln))
        if not os.path.isdir(ac_path):
            os.makedirs(ac_path)
        img_name = os.path.basename(img.getPath())
        shutil.copy(img.getPath(),os.path.join(ac_path,'{}.{}'.format(img_in,img_name.split('.')[1])))
        
    if not os.path.isdir(config.out_dir):
        os.mkdir(config.out_dir)

    if config.sdir is None or not os.path.isdir(config.sdir):
        print("Directory not found ({}), define -sd otion".format(config.sdir))
        sys.exit(1)

    files = os.listdir(config.sdir)

    acfiles = {}
    for f in files:
        if f.startswith('al-clustermetadata'):
            ac_id = int(f.split('.')[0].split('-')[3][1:])
            acfiles[ac_id] = os.path.join(config.sdir,f)

    ac_imgs = {}
    for k in config.ac_n:
        if not k in acfiles:
            print("Requested acquisition ({}) is not present".format(k))
            return None
        
        with open(acfiles[k],'rb') as fd:
            pool,un_clusters,un_indexes = pickle.load(fd)
    
        for cln in range(len(un_clusters)):
            ind = np.asarray(un_clusters[cln])
            print("Cluster {}, # of items: {}".format(cln,ind.shape[0]))
            posa = np.ndarray(shape=(1,),dtype=np.int32)
            for ii in range(min(ind.shape[0],config.n)):
                if ii == 0:
                    posa[0] = np.where(un_indexes == ind[ii])[0]
                else:
                    posa = np.hstack((posa,np.where(un_indexes == ind[ii])[0]))
                #Copy image
                _copy_img(config.out_dir,k,cln,pool[0][ind[ii]],posa[ii])
            print("Cluster {} first items positions in index array (at most {}): {}".format(cln,config.n,posa))
    
    
if __name__ == "__main__":


    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Convolunional Neural \
        Network for Image Segmentation.')

    parser.add_argument('--meta', dest='meta', action='store_true', 
        help='Acquire from ALTrainer metadata.', default=False)
    parser.add_argument('--cluster', dest='cluster', action='store_true', 
        help='Acquire from KM clustering metadata.', default=False)
    
    parser.add_argument('-ac', dest='ac_n', nargs='+', type=int, 
        help='Acquisitions to obtain images.', default=None,required=True)
    parser.add_argument('-sd', dest='sdir', type=str,default=None, 
        help='Experiment result path.')
    parser.add_argument('-od', dest='out_dir', type=str,default='img_grab', 
        help='Save selected images to this directory.')
    parser.add_argument('-n', dest='n', type=int, 
        help='Grab this many images. If cluster, grab this many images per cluster', default=200,required=False)

    config, unparsed = parser.parse_known_args()

    if config.meta:
        process_al_metadata(config)
    elif config.cluster:
        process_cluster_metadata(config)
    else:
        print("You should choose between ALTrainer metadata (--meta) of KM metadat (--cluster)")
