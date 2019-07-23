#!/usr/bin/env python3
#-*- coding: utf-8

import numpy as np
from scipy.stats import mode

def test_varratios(data_size,dp_steps,query,gen_random=True):
    All_Dropout_Classes = np.zeros(shape=(data_size,1))

    for d in range(dp_steps):
        print("Step {0}/{1}".format(d+1,dp_steps))
        if gen_random:
            proba = np.random.random((data_size,2))
        else:
            proba = np.zeros((data_size,2),dtype=np.float)
        dropout_classes = proba.argmax(axis=-1)
        dropout_classes = np.array([dropout_classes]).T
        All_Dropout_Classes = np.append(All_Dropout_Classes, dropout_classes, axis=1)

    Variation = np.zeros(shape=(data_size))

    print("All classes: {0}".format(All_Dropout_Classes))
    for t in range(data_size):
        L = np.array([0])
        for d_iter in range(dp_steps):
            L = np.append(L, All_Dropout_Classes[t, d_iter+1])

        print("Vetor L: {0}".format(L))
        Predicted_Class, Mode = mode(L[1:])
        print("Predicted_Class: {0}; Mode: {1}".format(Predicted_Class,Mode))
        v = np.array(  [1 - Mode/float(dp_steps)])
        Variation[t] = v

    print("Variation array: {0}".format(Variation))
    a_1d = Variation.flatten()
    print("Vetor a_1d: {0}".format(a_1d))
    a_1d_sorted = a_1d.argsort()
    print("Array a_1d sorted: {0}".format(a_1d_sorted))
    x_pool_index = a_1d_sorted[-query:][::-1]
    print("Indexes: {0}".format(x_pool_index))
    print("Sorted a_1d: {0}".format(a_1d[x_pool_index]))

if __name__ == "__main__":
    test_varratios(200,5,10,gen_random=False)
