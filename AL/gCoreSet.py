# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Returns points that minimizes the maximum distance of any point to a center.

Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017

Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).

Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from sklearn.metrics import pairwise_distances

from .Common import load_model_weights

def __flatten_X(X):
    shape = X.shape
    flat_X = X
    if len(shape) > 2:
      flat_X = np.reshape(X, (shape[0],np.product(shape[1:])))
    return flat_X


def __update_distances(cluster_centers, min_distances, features, already_selected,
                           metric='euclidean', only_new=True, reset_dist=False):
    """Update min distances given cluster centers.

    Args:
      cluster_centers: indices of cluster centers
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    if reset_dist:
      min_distances = None
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in already_selected]
    if cluster_centers:
      # Update min_distances for all examples given new cluster center.
      x = features[cluster_centers]
      dist = pairwise_distances(features, x, metric=metric)

      if min_distances is None:
        min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        min_distances = np.minimum(min_distances, dist)

    return min_distances

def __select_batch(already_selected, features, N, **kwargs):
    """
    Diversity promoting active learning method that greedily forms a batch
    to minimize the maximum distance to a cluster center among all unlabeled
    datapoints.
    
    Args:
      already_selected: index of datapoints already selected (list)
      N: batch size

    Returns:
      indices of points selected to minimize distance to cluster centers
    """

    if already_selected is None:
        already_selected = []
        
    min_distances = None
    
    if len(already_selected) > 0:
        min_distances = __update_distances(already_selected,
                                               min_distances=min_distances,
                                               features = features,
                                               already_selected = [],
                                               only_new=False,
                                               reset_dist=True)

    new_batch = []

    for _ in range(N):
        if already_selected is None:
            # Initialize centers with a randomly selected datapoint
            ind = np.random.choice(np.arange(X.shape[0]))
        else:
            ind = np.argmax(min_distances)
        # New examples should not be in already selected since those points
        # should have min_distance of zero to a cluster center.
        assert ind not in already_selected

        min_distances = __update_distances([ind],
                                               min_distances=min_distances,
                                               features = features,
                                               already_selected = new_batch,
                                               only_new=True,
                                               reset_dist=False)
        new_batch.append(ind)
    print('Maximum distance from cluster centers is %0.2f'
              % max(min_distances))

    return np.asarray(new_batch)


def core_set(bayesian_model,generator,data_size,**kwargs):
    """
    Cluster data in K clusters and returns their centroids

    Function needs to extract the following configuration parameters:
    model <None>: not used. Model used to extract features will be created here
    generator <keras.Sequence>: data generator for predictions
    data_size <int>: number of data samples
    config <argparse>: configuration object
    mc_dp <int>: number of dropout iterations
    cpu_count <int>: number of cpu cores (used to define number of generator workers)
    gpu_count <int>: number of gpus available
    verbose <int>: verbosity level
    pbar <boolean>: user progress bars
    sw_threads <thread Object>: if a thread object is passed, you must wait its conclusion before loading weights
    """
    from sklearn.decomposition import PCA
    import importlib
    import copy
    import time
    from datetime import timedelta
    from Utils import CacheManager

    cache_m = CacheManager()
    
    if 'config' in kwargs:
        config = kwargs['config']
        gpu_count = config.gpu_count
        cpu_count = config.cpu_count
        verbose = config.verbose
        pbar = config.progressbar
        query = config.acquire
        clusters = config.clusters
    else:
        return None
    
    if 'acquisition' in kwargs:
        acq = kwargs['acquisition']
    else:
        acq = config.acquisition_steps
        
    if 'model' in kwargs:
        model = kwargs['model']
    else:
        print("[core_set] GenericModel is needed by core_set. Set model kw argument")
        return None

    #Models that take to long to save weights might not have finished
    if 'sw_thread' in kwargs:
        last_thread = None
        if isinstance(kwargs['sw_thread'],list):
            last_thread = kwargs['sw_thread'][-1]
        else:
            last_thread = kwargs['sw_thread']
        if config.ffeat is None and last_thread.is_alive():
            if config.info:
                print("[core_set] Waiting for model weights to become available...")
            last_thread.join()
    elif config.info:
        print("[core_set] Weights thread not available...trying to load weights")

    if not model.is_ensemble() and not (os.path.isfile(model.get_weights_cache()) or not os.path.isfile(model.get_mgpu_weights_cache())):
        if config.info:
            print("[core_set] No trained model or weights file found (H5).")
        return None    

    #Run feature extraction and clustering
    if hasattr(model,'build_extractor'):
        single_m,parallel_m = model.build_extractor(training=False,feature=True,parallel=False)
    else:
        if config.info:
            print("[km_uncert] Model is not prepared to produce features. No feature extractor")
        return None

    if not model.is_ensemble():
        pred_model = load_model_weights(config,model,single_m,parallel_m)
    else:
        generator.set_input_n(config.emodels)
        if not parallel_m is None:
            pred_model = parallel_m
        else:
            pred_model = single_m
            
    #Extract features for all images in the pool
    if config.info:
        print("Starting feature extraction ({} batches)...".format(len(generator)))        
    features = pred_model.predict_generator(generator,
                                            workers=4*cpu_count,
                                            max_queue_size=100*gpu_count,
                                            verbose=0)

    del(pred_model)
    del(parallel_m)
    del(single_m)
        
    features = features.reshape(features.shape[0],np.prod(features.shape[1:]))

    if config.info:
        print("Feature vector shape: {}".format(features.shape))
            
    if config.pca > 0:
        if config.info:
            print("Starting PCA decomposition ({} features)...".format(config.pca))

        pca = PCA(n_components = config.pca)
        features = pca.fit_transform(features)

    stime = None
    etime = None
    if config.verbose > 0:
        print("Done extraction...starting CoreSet")
        stime = time.time()

    acquired = __select_batch(already_selected=[], features, query)

    print("Acquired ({}): {}".format(acquired.shape,acquired))

    return acquired
