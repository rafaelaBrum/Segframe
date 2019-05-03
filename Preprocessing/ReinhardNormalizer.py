#!/usr/bin/env python3
#-*- coding: utf-8
# From: https://github.com/SBU-BMI/quip_cnn_segmentation/tree/develop/large_scale_training_testing

from __future__ import division

import os
import numpy as np
from skimage import color,io

### Some functions ###


def lab_split(I):
    """
    Convert from RGB uint8 to LAB and split into channels
    :param I: uint8
    :return:
    """
    I = color.rgb2lab(I)
    I = I.astype(np.float32)
    I1, I2, I3 = I[0],I[1],I[2]
    I1 /= 2.55
    I2 -= 128.0
    I3 -= 128.0
    return I1, I2, I3


def merge_back(I1, I2, I3):
    """
    Take seperate LAB channels and merge back to give RGB uint8
    :param I1:
    :param I2:
    :param I3:
    :return:
    """
    I1 *= 2.55
    I2 += 128.0
    I3 += 128.0
    I = np.clip(np.dstack((I1, I2, I3)), 0, 255).astype(np.uint8)
    return color.lab2rgb(I)


def get_mean_std(I):
    """
    Get mean and standard deviation of each channel
    :param I: uint8
    :return:
    """
    I1, I2, I3 = lab_split(I)
    m1 = np.mean(I1)
    sd1 = np.std(I1)
    m2 = np.mean(I2)
    sd2 = np.std(I2)
    m3 = np.mean(I3)
    sd3 = np.std(I3)

    means = m1, m2, m3
    stds = sd1, sd2, sd3
    return means, stds

"""
Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""

def standardize_brightness(I):
    """

    :param I:
    :return:
    """
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)

### Main class ###

class Normalizer(object):
    """
    A stain normalization object
    """

    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target):
        target = standardize_brightness(target)
        means, stds = get_mean_std(target)
        self.target_means = means
        self.target_stds = stds

    def transform(self, I):
        I = standardize_brightness(I)
        I1, I2, I3 = lab_split(I)
        means, stds = get_mean_std(I)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / (1e-10+stds[0]))) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / (1e-10+stds[1]))) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / (1e-10+stds[2]))) + self.target_means[2]
        return merge_back(norm1, norm2, norm3)

class ReinhardNormalizer(object):
    def __init__(self, target_file):
        if isinstance(target_file,str) and os.path.isfile(target_file):
            target_40X = io.imopen(target_file)
        else:
            raise ValueError("[ReinhardNormalizer] Target file should be an image file")
        
        self.n_40X = Normalizer()
        self.n_40X.fit(target_40X)

    def normalize(self, image):
        # image RGB in uint8
        return self.n_40X.transform(image)
