import numpy as np
from scipy import ndimage as ndimg
import imageio
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import os,os.path 
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Conv2D,MaxPooling2D


base_model = VGG16(include_top = False, weights = "imagenet",input_shape=(64,64,3) )
w = base_model.output

w = Flatten()(w)
w = Dense(100, activation = "relu")(w)
w = Dense(2,activation = "relu")(w)

model = Model(inputs = [base_model.input], outputs = [w])
model.summary()

path1 = "./GTSRB_subset_2/class1"
path2 = "./GTSRB_subset_2/class2"

# load images from folders
images = []
target = []
norm_images = []


for file in os.listdir(path1):
    img = imageio.imread(os.path.join(path1, file))
    images.append(img)
    target.append('0')
    
for file in os.listdir(path2):
    img = imageio.imread(os.path.join(path2, file))
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

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.fit(X_train,y_train,nb_epoch=10,batch_size=32,validation_data=[X_test,y_test])

