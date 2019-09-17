#!/usr/bin/env python3
#-*- coding: utf-8
#Author: André L. S. Meirelles (andre.meirelles@aluno.unb.br)

import numpy as np
import os
import shutil
import sys
import argparse
import pickle

def grab_images(config):

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
            
    return [ac_imgs[k][:config.n] for k in config.ac_n]

if __name__ == "__main__":


    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Convolunional Neural \
        Network for Image Segmentation.')

    parser.add_argument('-ac', dest='ac_n', nargs='+', type=int, 
        help='Acquisitions to obtain images.', default=None,required=True)

    parser.add_argument('-sd', dest='sdir', type=str,default=None, 
        help='Experiment result path.')
    parser.add_argument('-od', dest='out_dir', type=str,default='img_grab', 
        help='Save selected images to this directory.')
    parser.add_argument('-n', dest='n', type=int, 
        help='Grab this many images.', default=200,required=False)

    config, unparsed = parser.parse_known_args()

    acquisitions = grab_images(config)

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

    print("Acquired images copied to output dir")
    
