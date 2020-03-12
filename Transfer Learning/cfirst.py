# from http://learnandshare645.blogspot.ie/2016/06/3d-cnn-in-keras-action-recognition.html
# Action recognition in Keras.
# On friday, modify this to work with our data.
import keras.backend as kb

kb.set_image_dim_ordering('tf')

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Convolution1D
from keras.optimizers import SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D

from keras.layers.recurrent import LSTM
import keras.backend as K


from keras.utils import np_utils, generic_utils

import matplotlib.pyplot as plt

labels=np.load('labelssmall.np.npy')
labelsVal=np.load('vallabelssmall.np.npy')
clips=np.load('clipssmall.np.npy')
clipsVal=np.load('valclipssmall.np.npy')

print np.shape(clips)

clipsVal=np.reshape(clipsVal,(len(clipsVal),1,5, 60, 80))
clips=np.reshape(clips,(len(clips),1,5, 60, 80))

Y_train = np_utils.to_categorical(labels)
Y_val=np_utils.to_categorical(labelsVal)

# Pre-processing

train_set = np.asarray(clips, dtype='float32')
train_set -= np.mean(train_set)
train_set /= np.max(train_set)

val_set = np.asarray(clipsVal, dtype='float32')
val_set -= np.mean(val_set)
val_set /= np.max(val_set)

X_train_new=train_set

X_val_new=val_set
y_train_new=Y_train
y_val_new =Y_val

#clips=np.transpose(clips,(0,1,4,2,3))

print np.shape(clips)

#clips=np.reshape(clips,(len(clips),5, 3, img_rows, img_cols))

# image specification

# Training data

batch_size = 10
nb_classes = 10
nb_epoch = 7


# Define model

model = Sequential()

#model.add(ConvLSTM2D(nb_filter=40,nb_col=3,nb_row=3, input_shape=(3, img_rows, img_cols, 5), return_sequences=True, border_mode='same'))

#5, 3, 480, 640
model.add(Convolution3D(16, 3, 3, 3,  activation='relu', input_shape=(1,5, 60, 80)))

model.add(ConvLSTM2D(nb_filter=10,nb_col=3,nb_row=3, return_sequences=False, border_mode='same'))

model.add(Flatten())
model.add(Dense(nb_classes, init='normal'))
model.add(Activation('softmax'))


sgd = SGD(lr=0.001, momentum=0.95)


model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"], )


print 'trained on small batch'

hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new, y_val_new),  batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True)
