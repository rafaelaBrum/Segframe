import numpy as np
import openslide
import sys
import os
from PIL import Image
import argparse
import multiprocessing

#Local functions
from Preprocessing import background,white_ratio

def make_tiles(slide_name,output_folder,patch_size_20X,wr):
    try:
        oslide = openslide.OpenSlide(slide_name);
        #mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
        if openslide.PROPERTY_NAME_MPP_X in oslide.properties:
            mag = 10.0 / float(oslide.properties[openslide.PROPERTY_NAME_MPP_X]);
        elif "XResolution" in oslide.properties:
            mag = 10.0 / float(oslide.properties["XResolution"]);
        elif 'tiff.XResolution' in oslide.properties:   # for Multiplex IHC WSIs, .tiff images
            mag = 10.0 / float(oslide.properties["tiff.XResolution"]);
        else:
            mag = 10.0 / float(0.254);
        pw = int(patch_size_20X * mag / 20);
        width = oslide.dimensions[0];
        height = oslide.dimensions[1];
    except:
        print('{}: exception caught'.format(slide_name));
        exit(1);


    pcount = 0
    print(slide_name, width, height);

    for x in range(1, width, pw):
        for y in range(1, height, pw):
            if x + pw > width:
                pw_x = width - x;
            else:
                pw_x = pw;
            if y + pw > height:
                pw_y = height - y;
            else:
                pw_y = pw;

            if (int(patch_size_20X * pw_x / pw) <= 0) or \
              (int(patch_size_20X * pw_y / pw) <= 0) or \
              (pw_x <= 0) or (pw_y <= 0):
                continue;

            patch = oslide.read_region((x, y), 0, (pw_x, pw_y));
            np_patch = np.array(patch)
            if not background(np_patch) and white_ratio(np_patch) <= wr:
                patch = patch.resize((int(patch_size_20X * pw_x / pw), int(patch_size_20X * pw_y / pw)), Image.ANTIALIAS);
                fname = '{}/{}_{}_{}_{}.png'.format(output_folder, x, y, pw, patch_size_20X);
                patch.save(fname);
                pcount += 1

    oslide.close()
    return pcount


if __name__ == "__main__":

    #Parse input parameters
    arg_groups = []
    parser = argparse.ArgumentParser(description='Extract tiles from a WSI \
        discarding background.')
        
    parser.add_argument('-ds', dest='ds', type=str,default='WSI', 
        help='Path to WSIs to tile (directory containing .svs images).')        
    parser.add_argument('-od', dest='out_dir', type=str, default='Patches', 
        help='Save extracted patches to this location.')
    parser.add_argument('-mp', dest='mp', type=int, 
        help='Use multiprocessing. Number of processes to spawn', default=2,required=False)
    parser.add_argument('-label', action='store_true', dest='label',
        help='Generate labels for the patches from heatmaps.',default=False)
    parser.add_argument('-hm', dest='heatmap', type=str,default=None, 
        help='Heatmaps path.')
    parser.add_argument('-ps', dest='patch_size', type=int, 
        help='Patch size in 20x magnification (Default 500)', default=500,required=False)
    parser.add_argument('-wr', dest='white', type=float, 
        help='Maximum white ratio allowed for each patch (Default: 0.20)', default=0.2,required=False)
    
    config, unparsed = parser.parse_known_args()

    if not os.path.exists(config.out_dir):
        os.mkdir(config.out_dir);

    if not os.path.isdir(config.ds):
        print("Path not found: {}".format(config.ds))
        sys.exit(1)

    wsis = os.listdir(config.ds)
    wsis = list(filter(lambda x:x.split('.')[-1] == 'svs',wsis))
    
    results = None
    total_patches = 0
    with multiprocessing.Pool(processes=config.mp) as pool:
        results = [pool.apply_async(make_tiles,(os.path.join(config.ds,i),config.out_dir,config.patch_size,config.white)) for i in wsis]
        total_patches = sum([r.get() for r in results])

    print("Total of patches generated: {}".format(total_patches))
