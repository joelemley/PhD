# from http://learnandshare645.blogspot.ie/2016/06/3d-cnn-in-keras-action-recognition.html
# Action recognition in Keras.
# On friday, modify this to work with our data.


import keras.backend as kb

kb.set_image_dim_ordering('th')


#1, img_rows, img_cols, patch_size

import numpy as np

#labels=np.load('labels.np.npy')
#clips=np.load('clips.np.npy')


labels=np.load('labelssmall.np.npy')
labelsVal=np.load('vallabelssmall.np.npy')

clips=np.load('clipssmall.np.npy')
clipsVal=np.load('valclipssmall.np.npy')

clipsVal=np.reshape(clipsVal,(len(clipsVal),1,30, 60, 80))
clips=np.reshape(clips,(len(clips),1,30, 60, 80))

#    e.g. `input_shape=(None,3, 10, 128, 128)` for 10 frames of 128x128 RGB pictures.



import matplotlib.pyplot as plt
img=clips[10000][0][1]
test=plt.imshow(img,cmap='gray')
plt.show()
print np.shape(clips)

#clips=np.transpose(clips,(0,1,3,4,2))
#clipsVal=np.transpose(clipsVal,(0,1,3,4,2))
print np.shape(clips)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Convolution1D
from keras.optimizers import SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM

from keras.utils import np_utils, generic_utils

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cross_validation import train_test_split

# image specification

# Training data

batch_size = 300
nb_classes = 10
nb_epoch = 20

Y_train = np_utils.to_categorical(labels)
Y_trainVal=np_utils.to_categorical(labelsVal)

# Pre-processing

train_set = np.asarray(clips, dtype='float32')
#train_set -= np.mean(train_set)
train_set /= np.max(train_set)

valset = np.asarray(clipsVal, dtype='float32')

#valset -= np.mean(valset)
valset /= np.max(valset)

# Define model

model = Sequential()

#
# model.add(Convolution3D(16, 3, 3, 3,input_shape=(5, 60, 80,1),  activation='relu'))
# model.add(Dropout(0.5))
# model.add(Convolution3D(16,3,3,3,activation='relu'))
# model.add(Dropout(0.5))
# model.add(ConvLSTM2D(nb_filter=10,nb_col=3,nb_row=3, return_sequences=False, border_mode='same'))
# model.add(Flatten())
# model.add(Dense(nb_classes, init='normal'))
# model.add(Activation('softmax'))


#model.add(ConvLSTM2D(nb_filter=4,nb_col=3,nb_row=3, return_sequences=False, border_mode='same',input_shape=(5, 60, 80,1)))


model.add(Convolution3D(8, 3, 3, 3,input_shape=(1,16, 60,80),  activation='relu'))
model.add(Dropout(p=.5))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(2,2,2)))
model.add(Convolution3D(8, 3, 3, 3,  activation='relu', border_mode='same'))
model.add(Dropout(p=.5))
model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(2,2,2)))
model.add(Convolution3D(8, 3, 3, 3,  activation='relu', border_mode='same'))
model.add(Dropout(p=.5))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(nb_classes, init='normal'))
model.add(Activation('softmax'))

sgd = Adam(lr=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"], )

# Split the data

###X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_set, Y_train, test_size=0.2, random_state=4)

print 'trained on small batch'

hist = model.fit(train_set, Y_train, validation_data=(valset, Y_trainVal),  batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True)

# hist = model.fit(train_set, Y_train, batch_size=batch_size,
#         nb_epoch=nb_epoch,validation_split=0.2, show_accuracy=True,
#           shuffle=True)


# Evaluate the model
#score = model.evaluate(X_val_new, y_val_new, batch_size=batch_size)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])

