# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 19:16:19 2018

@author: mital

Cifar-10 data set was collected by Alex Krizhevsky. For more information, see
Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.

Below utility loads the image data as numpy arrays and to test that data is 
loaded correctly a figure window is opened for 5 seconds (this can be changed 
by parameter)

Parameters
    ----------
    pause_inteval : integer, optional
                    This parameter governs the closing time for figure window. 
                    The figure window gives a snapshot of 20 images, 10 each
                    selected at random from training and test set. 

"""

def load_cifar_data(pause_inteval = 5):

    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
     
    # Cifra-10 classes for images
    cifar_10_cat = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    tr_data = np.zeros((50000,3072),dtype = 'uint8')
    tr_labels = []

    te_data = np.zeros((10000,3072),dtype = 'uint8')
    te_labels = []
    
    # List of files to be unpickled (deserialized)
    batch_list = ["C:\\Local\\mital\\Data\\Cifar_Data\\data\\data_batch_1","C:\\Local\\mital\\Data\\Cifar_Data\\data\\data_batch_2","C:\\Local\\mital\\Data\\Cifar_Data\\data\\data_batch_3","C:\\Local\\mital\\Data\\Cifar_Data\\data\\data_batch_4","C:\\Local\\mital\Data\\Cifar_Data\\data\\data_batch_5","C:\\Local\\mital\\Data\\Cifar_Data\\data\\test_batch"]
    # An empty list to store data dictionary
    dict_list = []

    for file in batch_list:
        print("Loading... " + file)
        with open(file, 'rb') as fo:
            dict_list.append(pickle.load(fo, encoding='bytes'))      


    for index in range(5):
        tr_data[10000*index:10000*(index + 1),:] = (dict_list[index])[b'data']
        tr_labels.extend((dict_list[index])[b'labels'])
    
    te_data = (dict_list[5])[b'data']
    te_labels.extend((dict_list[5])[b'labels']) 
    
    # Testing the data loaded by displaying some of the randomly selected images
    tr_random_index = np.random.randint(1,50000,10);
    te_random_index = np.random.randint(1,10000,10);
    
    fig = plt.figure()
    for i in range(len(tr_random_index)):
        img_vec = tr_data[tr_random_index[i],:] 
        
        img_r = np.reshape(img_vec[0:1024],(32,32))
        img_g = np.reshape(img_vec[1024:2048],(32,32))
        img_b = np.reshape(img_vec[2048:3072],(32,32))
    
        img = np.zeros((32,32,3),dtype = 'uint8');
    
        img[:,:,0] = img_r
        img[:,:,1] = img_g
        img[:,:,2] = img_b
        
        fig.add_subplot(1,10,i+1)
        plt.axis('off')        
        plt.title(cifar_10_cat[tr_labels[tr_random_index[i]]])
        plt.imshow(img)
        
    
    for i in range(len(te_random_index)):
        img_vec = te_data[te_random_index[i],:] 
        
        img_r = np.reshape(img_vec[0:1024],(32,32))
        img_g = np.reshape(img_vec[1024:2048],(32,32))
        img_b = np.reshape(img_vec[2048:3072],(32,32))
    
        img = np.zeros((32,32,3),dtype = 'uint8');
    
        img[:,:,0] = img_r
        img[:,:,1] = img_g
        img[:,:,2] = img_b
        
        fig.add_subplot(2,10,i+1)
        plt.axis('off') 
        plt.title(cifar_10_cat[te_labels[te_random_index[i]]])
        plt.imshow(img)     
        
    plt.show()    
    plt.pause(5)
    plt.close()
    return (tr_data,tr_labels,te_data,te_labels)







