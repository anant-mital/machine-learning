from scipy.io import loadmat
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


matData = loadmat("arcene.mat")

X_train = matData.get("X_train")
X_test = matData.get("X_test")
y_train = np.ravel(matData.get("y_train"))
y_test = np.ravel(matData.get("y_test"))

clf = LogisticRegression(penalty='l1')
C_range = 10.0 ** np.arange(0,12,0.25)

accuracies = []
nonzeros = []

for C in C_range:
    clf.C = C
    clf.fit(X_train,y_train)
    prediction = clf.predict(X_test)
    accuracy = 100.0 * np.mean(prediction == y_test)
    print("-----------------------------")
    print(C)
    print(accuracy)
    print(np.count_nonzero(clf.coef_))
    print("-----------------------------")
    accuracies.append(accuracy)
    


