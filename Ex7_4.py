from scipy.io import loadmat
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score


matData = loadmat("arcene.mat")

X_train = matData.get("X_train")
X_test = matData.get("X_test")
y_train = np.ravel(matData.get("y_train"))
y_test = np.ravel(matData.get("y_test"))

clf = LogisticRegression(penalty='l1')

rfe = RFE(clf,step=50,verbose=1)
rfe.fit(X_train,y_train)
print(accuracy_score(y_test, rfe.predict(X_test)))

