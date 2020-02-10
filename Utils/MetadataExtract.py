#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

import numpy as np
import os,sys
import shutil
import math
import argparse
import pickle
import importlib
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
            mask = np.isin(list(train[0]),initial_set,assume_unique=True,invert=True)
            imgs = train[0][mask]
            labels = train[1][mask]
            print("Acquired {} images in acquisition {}".format(imgs.shape[0],k-1))
            ac_imgs[k-1] = (imgs,labels)
            initial_set = list(train[0])    

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
    wsi_mean = 0.0
    for c in range(config.nc):
        idx = np.where(km.labels_ == c)[0]
        count_m = 0
        c_count = 0
        mean_d = 0.0
        print("Cluster {} center: {}".format(c,km.cluster_centers_[c]))
        x2 = km.cluster_centers_[c]
        for p in idx:
            x1 = members[p].getCoord()
            if x1 is None:
                continue
            c_count += 1
            dist = math.sqrt((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2)
            mean_d += dist
            if dist < config.radius:
                distances[members[p]] = dist
                count_m += 1
        print("    - patches in cluster: {}".format(idx.shape[0]))
        print("    - {} patches are within {} pixels from cluster center".format(count_m,config.radius))
        if c_count > 0:
            mean_d = mean_d / c_count
            wsi_mean += mean_d
            print("    - {:2.2f} % of cluster patches are within range".format(100*count_m/c_count))
        print("    - Mean distance from cluster center: {:.1f}".format(mean_d))
        wrad += count_m        
    wsi_mean = wsi_mean/config.nc
    print("{:2.2f}% of all WSI patches are within cluster center ranges".format(100*wrad/len(members)))
    print("Mean distance of clusterd patches to centers: {:.1f}".format(wsi_mean))

    return wsi_mean

def _combine_acquired_ds(wsis,ds_wsis,max_w=20):
    """
    wsis and ds_wsis: dictionary of tuples:
    wsi[k] -> ([imgs],[labels])
    k -> WSI name
    """
    acquired = list(wsis.keys())
    acquired.sort(key=lambda x:len(wsis[x][0]))

    acquired = acquired[:max_w]

    total_patches = 0
    for d in ds_wsis:
        total_patches += len(ds_wsis[d][0])

    print("\n"*3 + " "*10 + "CHECKING ACQUIRED IMAGES AGAINST DATASET")
    print(" "*10 + "Total of patches in dataset: {}".format(total_patches))
    for w in acquired:
        acq = len(wsis[w][0])
        dsp = len(ds_wsis[w][0])
        print("WSI {}:".format(w))
        print("   - {} patche(s) acquired".format(acq))
        print("   - {:2.4f} % of available patches from the WSI ({})".format(100*acq/dsp,dsp))
        print("   - {:2.4f} % of dataset patches are from this WSI".format(100*dsp/total_patches))
        
def _save_wsi_stats(wsis,dest,ac_save):

    init_k = 1
    for a in ac_save:
        save_to = os.path.join(dest,'patch_coordinates-{}-{}.txt'.format(init_k,a))
        fd = open(save_to,'w')
        fd.write('X-coordinate Y-coordinate Width Height\n')
        patches = {}
        for k in range(init_k,a):
            for w in wsis[k]:
                patches.setdefault(w,[])
                patches[w].extend(wsis[k][w])

        for ws in patches:
            fd.write('WSI {}: {}\n'.format(ws,len(patches[ws])))
            for i in patches[ws]:
                coord = i.getCoord()
                if not coord is None:
                    try:
                        dim = i.getImgDim()
                    except FileNotFoundError:
                        dim = (0,0)
                    fd.write('{0} {1} {2} {3}\n'.format(coord[0],coord[1],dim[0],dim[1]))
                else:
                    fd.write('No coordinates\n')
        init_k = a
        fd.close()

def _generate_label_files(config):
    """
    This should be run after process_al_metadata
    """
    dirs = os.listdir(config.out_dir)

    labels = {}
    for d in dirs:
        if not d in labels:
            ol = open(os.path.join(config.cp_orig,d,'label.txt'),'r')
            labels[d] = {}
            for line in ol.readlines():
                fields = line.strip().split(' ')
                labels[d][fields[0]] = fields[1:]
            ol.close()
        copy_to = os.path.join(config.out_dir,d)
        if os.path.isfile(os.path.join(copy_to,'label.txt')):
            fd = open(os.path.join(copy_to,'label.txt'),'a')
        else:
            fd = open(os.path.join(copy_to,'label.txt'),'w')
        for i in os.listdir(copy_to):
            if i.endswith('png'):
                fd.write('{} {}\n'.format(i,' '.join(labels[d][i])))

        fd.close()


def _append_label(config):
    """
    This should be run after process_al_metadata
    """
    dirs = os.listdir(config.out_dir)

    labels = {}
    for d in dirs:
        if not d in labels:
            ol = open(os.path.join(config.cp_orig,d,'label.txt'),'r')
            labels[d] = {}
            for line in ol.readlines():
                fields = line.strip().split(' ')
                labels[d][fields[0]] = fields[1] if int(fields[1]) >= 0 else '0'
            ol.close()
        move_to = os.path.join(config.out_dir,d)
        for i in os.listdir(move_to):
            if i.endswith('png'):
                nn = i.split('.')
                nn[0] += '_{}'.format(labels[d][i])
                shutil.move(os.path.join(move_to,i),os.path.join(move_to,'.'.join(nn)))
                
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
    
    if config.ac_n is None or config.save:
        #Use all data if specific acquisitions were not defined
        config.ac_n = list(ac_imgs.keys())
        config.ac_n.sort()

    if config.save and config.ac_save is None:
        config.ac_save = [min(config.ac_n),max(config.ac_n)]
        
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
                #continue

            if origin.startswith('log.'):
                origin = origin.split('.')[1]
            
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

    if config.save:
        _save_wsi_stats(acquisitions,config.sdir,config.ac_save)
        
    print("\n"+" "*10+"ACQUIRED PATCHES STATISTICS\n\n")
    #This dict will store, for each WSI, [#positive patches acquired, #total of positive patches]
    pos_patches = {}
    wsi_means = []
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
        if config.nc > 0:
            features = []
            for p in range(n_patches):
                if not wsis[s][0][p].getCoord() is None:
                    features.append(wsis[s][0][p].getCoord())
            features = np.asarray(features)
            if features.shape[0] > config.minp:
                km = KMeans(n_clusters = config.nc, init='k-means++',n_jobs=2).fit(features)
                wsi_means.append(_process_wsi_cluster(km,s,wsis[s][0],config))

    print("-----------------------------------------------------")
    print("Total of acquired patches: {}".format(total_patches))
    print("Total of positive patches acquired: {} ({:2.2f}%)".format(total_pos,100*total_pos/total_patches))
    print("WSIs used in acquisitions: {}".format(len(wsis)))
    if config.nc > 0:
        print("Acquired patches dispersion around cluster means: {:.1f}".format(np.mean(wsi_means)))

    #Generate dataset stats
    total_patches = 0
    discarded = 0
    total_pos = 0
    ds_wsis = {}
    print("\n"+" "*10+"DATASET PATCHES STATISTICS")
    if not config.cache_file is None:
        with open(config.cache_file,'rb') as fd:
            X,Y,_ = pickle.load(fd)
        ac_patches = len(X)
        for ic in range(ac_patches):
            img = X[ic]
            label = Y[ic]
            
            if hasattr(img,'getOrigin'):
                origin = img.getOrigin()
            elif hasattr(img,'_origin'):
                origin = img._origin
            else:
                print("Image has no origin information: {}".format(img.getPath()))
                continue
            
            if img.getCoord() is None:
                discarded += 1
                #continue
                
            if origin.startswith('log.'):
                origin = origin.split('.')[1]
                
            if origin in ds_wsis:
                ds_wsis[origin][0].append(img)
                ds_wsis[origin][1].append(label)
            else:
                ds_wsis[origin] = ([img],[label])
        

        for s in ds_wsis:
            n_patches = len(ds_wsis[s][0])
            labels = np.asarray(ds_wsis[s][1])
        
            #Count positive patches:
            unique,count = np.unique(labels,return_counts=True)
            l_count = dict(zip(unique,count))
            if 1 in l_count:
                if s in pos_patches:
                    pos_patches[s].append(l_count[1])
                    total_pos += l_count[1]
                else:
                    pos_patches[s] = [0,l_count[1]]
            else:
                if s in pos_patches:
                    pos_patches[s].append(0)
                else:
                    pos_patches[s] = [0,0]
            
            print("******   {} ({} total patches)  *******".format(s,n_patches))
            print("Positive patches: {} ({:2.2f}%)".format(pos_patches[s][1],100*pos_patches[s][1]/n_patches))
            if pos_patches[s][1] > 0:
                print("Positive patches acquired from this WSI: {} ({:2.2f}%)".format(pos_patches[s][0],100*pos_patches[s][0]/pos_patches[s][1]))
            else:
                print("Positive patches acquired from this WSI: {} (0.0%)".format(pos_patches[s][0]))
            if config.nc > 0:
                features = []
                for p in range(n_patches):
                    if not ds_wsis[s][0][p].getCoord() is None:
                        features.append(ds_wsis[s][0][p].getCoord())
                features = np.asarray(features)
                if features.shape[0] > config.minp:
                    km = KMeans(n_clusters = config.nc, init='k-means++',n_jobs=2).fit(features)
                    _process_wsi_cluster(km,s,ds_wsis[s][0],config)
        print("-----------------------------------------------------")
        print("Total patches in dataset: {}".format(ac_patches))
        print("Total of positive patches in dataset: {} ({:2.2f}%)".format(total_pos,100*total_pos/ac_patches))
        print("WSIs in dataset: {}".format(len(ds_wsis)))

        if config.comb_wsi:
            _combine_acquired_ds(wsis,ds_wsis)
            
def process_al_metadata(config):
    def change_root(s,d):
        """
        s -> original path
        d -> change location to d
        """
        components = tuple(s.split(os.path.sep)[-2:])
        relative_path = os.path.join(*components)

        return os.path.join(d,relative_path)
    
    if not os.path.isdir(config.out_dir):
        os.makedirs(config.out_dir)

    ac_imgs = _process_al_metadata(config)

    if config.ac_n is None:
        #Use all data if specific acquisitions were not defined
        config.ac_n = list(ac_imgs.keys())
        config.ac_n.sort()    
    
    acquisitions = [ac_imgs[k][0][:config.n] for k in config.ac_n]

    print("# of acquisitions obtained: {} -> ".format(len(acquisitions)),end='')
    print(" ".join([str(len(d)) for d in acquisitions]))
    print("Copying...")

    same_name = {}
    for a in range(len(config.ac_n)):
        ac_path = os.path.join(config.out_dir,str(config.ac_n[a]))
        
        if not os.path.isdir(ac_path) and not config.keep:
            os.mkdir(ac_path)
            
        for img in acquisitions[a]:
            img.setPath(change_root(img.getPath(),config.cp_orig))
            if not config.keep:
                img_name = os.path.basename(img.getPath())
                if img_name in same_name:
                    print("Images with the same name detected: {}\n{}\n{}".format(img_name,img.getPath(),same_name[img_name]))
                else:
                    same_name[img_name] = img.getPath()
                shutil.copy(img.getPath(),ac_path)
            else:
                cur_dir = os.path.split(os.path.dirname(img.getPath()))[-1]
                copy_to = os.path.join(config.out_dir,cur_dir)
                if not os.path.isdir(copy_to):
                    os.mkdir(copy_to)
                shutil.copy(img.getPath(),copy_to)

    if config.gen_label:
        print("Generating label files...")
        _generate_label_files(config)
    elif config.add_label:
        print("Appending label to file names...")
        _append_label(config)
        
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
        
    parser.add_argument('-ac', dest='ac_n', nargs='+', type=int,
        help='Acquisitions to obtain images.', default=None, required=False)
    parser.add_argument('-sd', dest='sdir', type=str,default=None, 
        help='Experiment result path.')
    parser.add_argument('-ds', dest='ds', type=str,default='CellRep', 
        help='Dataset used in experiments.')        
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
    parser.add_argument('-cache_file', dest='cache_file', type=str,default=None, 
        help='Dataset metadata for WSI statistics.')
    parser.add_argument('-save', action='store_true', dest='save',
        help='Saves generated statistics to file.',default=False)    
    parser.add_argument('-ac_save', dest='ac_save', nargs='+', type=int,
        help='Save image statistics from this this acquisitions.', default=None, required=False)
    parser.add_argument('-keep', action='store_true', dest='keep',
        help='Keep original dataset structure when copying tiles.',default=False)    
    parser.add_argument('-gen_label', action='store_true', dest='gen_label',
        help='Generate label file for extracted patches.',default=False)
    parser.add_argument('-add_label', action='store_true', dest='add_label',
        help='Append label to file names as used in quip_classification.',default=False)
    parser.add_argument('-comb_wsi', action='store_true', dest='comb_wsi',
        help='Check patches acquired from a WSI in comparison to total WSI patches available .',default=False)
    
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
