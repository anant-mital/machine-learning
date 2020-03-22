# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 23:54:06 2018

@author: mital

This function returns class of an input vector, by just giving a 
random value between [0,9].

Parameters
    ----------
    x : input numpy integer array
             
Returns
    ----------
    label : Integer value
            Between 0-9, these digits represents 10 classes of Cifar-10 dataset

"""

def cifar_10_1NN(x,tr_data,tr_labels):
    
    from numpy.matlib import repmat
    import numpy as np
    from scipy.spatial import distance
    
    shape_tr_data = tr_data.shape
    
    # now repeating matrix along the axes 0 (vertically down)
    # till the sample becomes matrix of same shape as the tr_data
    
    x = np.reshape(x,(1,shape_tr_data[1]))
    x_repmat = repmat(x,shape_tr_data[0],1)
    dist = distance.cdist(x_repmat,tr_data,'euclidean')
    
    
    
    
    return label