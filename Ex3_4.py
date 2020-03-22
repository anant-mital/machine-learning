from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()
print(digits.keys())

plt.gray()
plt.imshow(digits.images[0])
plt.show()

x_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size = 0.3)
clf = KNeighborsClassifier()
clf.fit(x_train,y_train)
print(accuracy_score(y_test, clf.predict(x_test)))