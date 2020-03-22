
import numpy as np
from scipy import ndimage as ndimg
from scipy import misc
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import os,os.path 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation

#
path1 = "./GTSRB_subset_2/class1"
path2 = "./GTSRB_subset_2/class2"

# load images from folders
images = []
target = []
norm_images = []


for file in os.listdir(path1):
    img = misc.imread(os.path.join(path1, file))
    images.append(img)
    target.append('0')
    
for file in os.listdir(path2):
    img = misc.imread(os.path.join(path2, file))
    images.append(img)
    target.append('1')
    
for image in images:
    norm_image = np.divide(image - np.min(image),float(np.max(image)))
    norm_images.append(norm_image)
    
np_images = np.asarray(norm_images)
np_labels = np.asarray(target)   

bin_labels = np_utils.to_categorical(np_labels,2)
    
X_train, X_test, y_train, y_test = train_test_split(
    np_images, bin_labels, test_size=0.2)   


N = 32
w, h = 5, 5

model = Sequential()

model.add(Conv2D(N,(w,h),input_shape=(64,64,3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Conv2D(N,(w,h), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Flatten())
model.add(Dense(100,activation='sigmoid'))
model.add(Dense(2,activation='sigmoid'))

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.fit(X_train,y_train,nb_epoch=20,batch_size=32,validation_data=[X_test,y_test])



    
 
    
    
    
    
    
    
    
    
    
    
    
    