# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 22:44:22 2018

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

def cifar_10_rand(x):
    
    import random;
    
    label = random.randint(0,9)
    return label