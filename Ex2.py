# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 19:16:19 2018
@author: mital

"""
import LoadCifarData
import RandomClassifier
import Evaluate


# Load Cifar Data
tr_data,tr_labels,te_data,te_labels = LoadCifarData.load_cifar_data(pause_inteval = 15)

# Run random classfier on test samples
pred = []

for ind in range(len(te_data)):
    category = RandomClassifier.cifar_10_rand(te_data[ind])
    pred.append(category)

clff_accuracy = Evaluate.cifar_10_evaluate(pred,te_labels)
print(clff_accuracy)
    
pred.clear()

# Nearest Neighbour
    






