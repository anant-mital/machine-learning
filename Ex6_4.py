from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D,MaxPooling2D
import numpy as np


N = 32
w, h = 3, 3

model = Sequential()

model.add(Conv2D(N,(w,h),input_shape=(128,128,3), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(4,4)))

model.add(Conv2D(48,(w,h), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(w,h), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))

