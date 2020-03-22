
import numpy as np
from scipy import ndimage as ndimg
from scipy import misc
from skimage.feature import local_binary_pattern as lbp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import os,os.path 

#
path1 = "./GTSRB_subset/class1"
path2 = "./GTSRB_subset/class2"

# load images from folders
images = []
target = []



for file in os.listdir(path1):
    img = misc.imread(os.path.join(path1, file))
    images.append(img)
    target.append('c1')
    
for file in os.listdir(path2):
    img = misc.imread(os.path.join(path2, file))
    images.append(img)
    target.append('c2')

features = []
METHOD = 'uniform'
radius = 3
n_points = 8 * radius

# extract features
for img in images:
    img = lbp(img, n_points, radius, METHOD)
    features.append(np.histogram(img)[0])
    
# create list of 3 classifiers

classifiers = [KNeighborsClassifier(),LinearDiscriminantAnalysis(),SVC()]   
 
# splitiing the data into cross validation set

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2)   

# training the classfiers and printing the results

for clf in classifiers:
    clf.fit(X_train,y_train)
    print(clf.__class__.__name__)
    print(accuracy_score(y_test, clf.predict(X_test)))   


    
 
    
    
    
    
    
    
    
    
    
    
    
    