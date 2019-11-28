#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

import numpy as np
import os,sys
import shutil
import math
import argparse
import pickle
from sklearn.cluster import KMeans
    
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
            ac_imgs[k-1] = (imgs,train[1])
            initial_set = list(train)    

    return ac_imgs

def _process_wsi_cluster(km,s,members,config):
    """
    Generate statistics for each cluster.
    km -> KMeans instance
    s -> WSI name
    members -> cluster members (img,label)
    config -> configurations
    """
    distances = {}
    wrad = 0
    for c in range(config.nc):
        idx = np.where(km.labels_ == c)[0]
        count_m = 0
        print("Cluster {} center: {}".format(c,km.cluster_centers_[c]))
        for p in idx:
            x1 = members[p].getCoord()
            x2 = km.cluster_centers_[c]
            dist = math.sqrt((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2)
            if dist < config.radius:
                distances[members[p]] = dist
                count_m += 1
        print("    - patches in cluster: {}".format(idx.shape[0]))
        print("    - {} patches are within {} pixels from cluster center".format(count_m,config.radius))
        wrad += count_m
        print("    - {:2.2f} % of cluster patches are within range".format(100*count_m/idx.shape[0]))
    print("{:2.2f}% of all WSI patches are within cluster center ranges".format(100*wrad/len(members)))

def process_wsi_metadata(config):
    """
    Metadata should contain information about the WSI that originated the patch
    """
    ac_imgs = _process_al_metadata(config)
    
    wsis = {}
    acquisitions = {}
    total_patches = 0
    total_pos = 0
    discarded = 0
    
    if config.ac_n[0] == -1:
        #Use all data if specific acquisitions were not defined
        config.ac_n = list(ac_imgs.keys())
        config.ac_n.sort()
        
    for k in config.ac_n:
        print("In acquisition {}:\n".format(k)) 
        acquisitions[k] = {}
        
        if not k in ac_imgs:
            continue

        ac_patches = len(ac_imgs[k][0])
        total_patches += ac_patches
        for ic in range(ac_patches):
            img = ac_imgs[k][0][ic]
            label = ac_imgs[k][1][ic]
            
            if hasattr(img,'getOrigin'):
                origin = img.getOrigin()
            elif hasattr(img,'_origin'):
                origin = img._origin
            else:
                print("Image has no origin information: {}".format(img.getPath()))
                continue
            
            if img.getCoord() is None:
                discarded += 1
                continue
            
            if origin in wsis:
                wsis[origin][0].append(img)
                wsis[origin][1].append(label)
            else:
                wsis[origin] = ([img],[label])
                
            if origin in acquisitions[k]:
                acquisitions[k][origin].append(img)
            else:
                acquisitions[k][origin] = [img]

        for w in acquisitions[k]:
            coords = [str(p.getCoord()) for p in acquisitions[k][w]]
            print(' '*3 + '**{} ({} patches):{}'.format(w,len(coords),''.join(coords)))

    print("{} patches were disregarded for not having coordinate data".format(discarded))

    print("\n"+" "*10+"ACQUIRED PATCHES STATISTICS")
    #This dict will store, for each WSI, [#positive patches acquired, #total of positive patches]
    pos_patches = {}
    for s in wsis:
        n_patches = len(wsis[s][0])
        labels = np.asarray(wsis[s][1])
        
        #Count positive patches:
        unique,count = np.unique(labels,return_counts=True)
        l_count = dict(zip(unique,count))
        if 1 in l_count:
            pos_patches[s] = [l_count[1]]
            total_pos += l_count[1]
        else:
            pos_patches[s] = [0]
            
        print("******   {} ({} total patches)  *******".format(s,n_patches))
        print("Positive patches acquired: {} ({:2.2f}%)".format(pos_patches[s][0],100*pos_patches[s][0]/n_patches))
        if config.nc > 0 and n_patches > config.minp:
            features = np.zeros((n_patches,2))
            for p in range(n_patches):
                features[p] = np.asarray(wsis[s][0][p].getCoord())
            km = KMeans(n_clusters = config.nc, init='k-means++',n_jobs=2).fit(features)
            _process_wsi_cluster(km,s,wsis[s][0],config)    

    print("-----------------------------------------------------")
    print("Total of acquired patches: {}".format(total_patches))
    print("Total of positive patches acquired: {} ({:2.2f}%)".format(total_pos,100*total_pos/total_patches))
    print("WSIs used in acquisitions: {}".format(len(wsis)))

def process_al_metadata(config):
    if not os.path.isdir(config.out_dir):
        os.mkdir(config.out_dir)

    ac_imgs = _process_al_metadata(config)
    
    acquisitions = [ac_imgs[k][0][:config.n] for k in config.ac_n]

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
        for i in train[0]:
            if i in trainsets:
                trainsets[i] += 1
            else:
                trainsets[i] = 1

    miss_count = 0
    hit_count = 0
    for j in trainsets:
        if trainsets[j] == 1:
            print("Image {} occurs in only one of the sets".format(j))
            miss_count += 1
        elif trainsets[j] == 2:
            hit_count += 1

    print("{} patches were found in only one set.".format(miss_count))
    print("{} patches were in both sets".format(hit_count))
    
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
        
    parser.add_argument('-ac', dest='ac_n', nargs='?', type=int, const=[-1],
        help='Acquisitions to obtain images.', default=None, required=True)
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
    parser.add_argument('-minp', dest='minp', type=int, 
        help='Cluster patches if WSI had this number of selected patches', default=10,required=False)
    parser.add_argument('-nc', dest='nc', type=int, 
        help='Number of clusters to group patches of each WSI', default=0,required=False)
    parser.add_argument('-radius', dest='radius', type=int, 
        help='Radius in pixels to group patches in each cluster.', default=300,required=False)
    parser.add_argument('-cache_file', dest='cache_file', type=str,default='CellRep-metadata.pik', 
        help='Dataset metadata for WSI statistics.')
    
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
