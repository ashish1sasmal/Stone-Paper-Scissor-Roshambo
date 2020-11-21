# @Author: ASHISH SASMAL <ashish>
# @Date:   21-11-2020
# @Last modified by:   ashish
# @Last modified time: 21-11-2020

import time

import cv2
import numpy as np

start_time = time.time()
from keras_squeezenet import SqueezeNet
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.layers import Activation, Dropout, Convolution2D, GlobalAveragePooling2D
from keras.models import Sequential
import tensorflow as tf
import os

print(f"[ Importing libraries time :: {time.time()-start_time}]")


IMG_SAVE_PATH = 'Images'

"""
def getModel():


    return model
"""


dataset = []
for dir in os.listdir(IMG_SAVE_PATH):
    path = os.path.join(IMG_SAVE_PATH, dir)
    for item in os.listdir(path):
        img = cv2.imread(os.path.join(path, item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (227, 227))
        dataset.append([img, dir])

CLASS_MAP = {
    "stone": 0,
    "paper": 1,
    "scissor": 2,
    "none": 3
}

def mapper(val):
    return CLASS_MAP[val]

print(len(dataset))

data , labels = zip(*dataset)
labels = list(map(mapper, labels))  # mapper is a function ; labels is to be mapped by mapper function

"""
A one hot encoding is a representation of categorical variables as binary vectors.
 This first requires that the categorical values be mapped to integer values.
 Then, each integer value is represented as a binary vector that is all zero values
 except the index of the integer, which is marked with a 1.
"""

labels = np_utils.to_categorical(labels)

model = Sequential([
    SqueezeNet(input_shape=(227, 227, 3), include_top=False),
    Dropout(0.5),
    Convolution2D(4, (1, 1), padding='valid'),
    Activation('relu'),
    GlobalAveragePooling2D(),
    Activation('softmax')
])

model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# start training
model.fit(np.array(data), np.array(labels), epochs=10)

# save the model for later use
model.save("stone-paper-scissors-model.h5")
