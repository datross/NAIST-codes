# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 11:01:22 2017

@author: mael
"""


################################################################## IMPORT

import cv2
import numpy as np
np.random.seed(123) # for reproducibility
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten # core layers
from keras.layers import convolutional, pooling # CNN layers
from keras.utils import np_utils



def loadImg(path):
    img = cv2.imread(path)
    if(img == None):
        print("Failed to load: ", path)
    return img[:,:,::-1]

img = loadImg("D:\\FCN segmentation\\assets\\miscellaneous\\in.jpg")


################################################################## BUILD NETWORK

# 7. Define model architecture
model = Sequential()
 
model.add(convolutional.Conv2D(32, (3, 3), activation='relu', input_shape=(None,None,3)))
model.add(convolutional.Conv2D(32, (3, 3), activation='relu'))
model.add(pooling.MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
 
## 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

img = img[None,...]

out = model.predict(img, verbose=1)

print(out.shape)

