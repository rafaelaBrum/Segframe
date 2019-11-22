#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

import numpy as np
import os
import shutil
import sys
import argparse
import pickle

def _process_al_metadata(config):
    """
    Returns the images acquired in each acquisition, as stored in al-metadata files
    """
    if config.sdir is None or not os.path.isdir(config.sdir):
        print("Directory not found: {}".format(config.sdir))
        sys.exit(1)

    files = os.listdir(config.sdir)

    acfiles = {}
    for f in files:
        if f.startswith('al-metadata'):
            ac_id = int(f.split('.')[0].split('-')[3][1:])
            net = f.split('.')[0].split('-')[2]
            if (not config.net is None and config.net == net) or config.net is None:
                acfiles[ac_id] = os.path.join(config.sdir,f)

    print(acfiles)
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

    return ac_imgs

def process_wsi_metadata(config):
    """
    Metadata should contain information about the WSI that originated the patch
    """
    ac_imgs = _process_al_metadata(config)
    
    wsis = {}
    for k in config.ac_n:
        print("In acquisition {}:\n".format(k)) 
        wsis[k] = {}
        
        if not k in ac_imgs:
            continue
        for img in ac_imgs[k]:
            if hasattr(img,'getOrigin'):
                origin = img.getOrigin()
            elif hasattr(img,'_origin'):
                origin = img._origin
            else:
                print("Image has no origin information: {}".format(img.getPath()))
                continue
            if origin in wsis[k]:
                wsis[k][origin].append(img)
            else:
                wsis[k][origin] = [img]

        for w in wsis[k]:
            coords = [str(p._coord) for p in wsis[k][w]]
            print(' '*3 + '**{} ({} patches):{}'.format(w,len(coords),''.join(coords)))
    
    
def process_al_metadata(config):
    if not os.path.isdir(config.out_dir):
        os.mkdir(config.out_dir)

    ac_imgs = _process_al_metadata(config)
    
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
    def _copy_img(dst_dir,acq,cln,img,img_in,orig=None):
        ac_path = os.path.join(dst_dir,str(acq),str(cln))
        if not os.path.isdir(ac_path):
            os.makedirs(ac_path)
        img_name = os.path.basename(img.getPath())
        if orig is None:
            shutil.copy(img.getPath(),os.path.join(ac_path,'{}.{}'.format(img_in,img_name.split('.')[1])))
        else:
            subdir = os.path.split(os.path.dirname(img.getPath()))[1]
            orig_img = os.path.join(orig,subdir,os.path.basename(img.getPath()))
            shutil.copy(orig_img,os.path.join(ac_path,'{}.{}'.format(img_in,img_name.split('.')[1])))
            
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
                _copy_img(config.out_dir,k,cln,pool[0][ind[ii]],posa[ii],config.cp_orig)
            print("Cluster {} first items positions in index array (at most {}): {}".format(cln,config.n,posa))
    

def process_train_set(config):

    trainsets = {}
    for f in config.trainset:
        if not os.path.isfile(f):
            print("File not found: {}".format(f))
            return None
        with open(f,'rb') as fd:
            train,_,_ = pickle.load(fd)
        for i in train:
            if i in trainsets:
                trainsets[i] += 1
            else:
                trainsets[i] = 1

    for j in trainsets:
        if trainsets[j] == 1:
            print("Image {} occurs in only one of the sets".format(j))
    
if __name__ == "__main__":


    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Convolunional Neural \
        Network for Image Segmentation.')

    parser.add_argument('--meta', dest='meta', action='store_true', 
        help='Acquire from ALTrainer metadata.', default=False)
    parser.add_argument('--cluster', dest='cluster', action='store_true', 
        help='Acquire from KM clustering metadata.', default=False)
    parser.add_argument('--wsi', dest='wsi', action='store_true', 
        help='Identify the patches of each WSI in the acquisitions.', default=False)
    parser.add_argument('--train_set', dest='trainset', type=str, nargs=2,
        help='Check if the training sets of two experiments are the same.', default=None)
        
    parser.add_argument('-ac', dest='ac_n', nargs='+', type=int, 
        help='Acquisitions to obtain images.', default=None,required=True)
    parser.add_argument('-sd', dest='sdir', type=str,default=None, 
        help='Experiment result path.')
    parser.add_argument('-od', dest='out_dir', type=str,default='img_grab', 
        help='Save selected images to this directory.')
    parser.add_argument('-n', dest='n', type=int, 
        help='Grab this many images. If cluster, grab this many images per cluster', default=200,required=False)
    parser.add_argument('-orig', dest='cp_orig', type=str, nargs='?', default=None, const='../data/lym_cnn_training_data',
        help='Copy original images instead of the normalized ones. Define location of the originals.')
    parser.add_argument('-net', dest='net', type=str,default=None, 
        help='Network name for metadata analysis.')
    
    config, unparsed = parser.parse_known_args()

    if config.meta:
        process_al_metadata(config)
    elif config.cluster:
        process_cluster_metadata(config)
    elif config.wsi:
        process_wsi_metadata(config)
    elif not config.trainset is None:
        process_train_set(config)
    else:
        print("You should choose between ALTrainer metadata (--meta), KM metadat (--cluster) or WSI metadata (--wsi)")
