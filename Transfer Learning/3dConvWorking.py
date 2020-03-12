# from http://learnandshare645.blogspot.ie/2016/06/3d-cnn-in-keras-action-recognition.html
# Action recognition in Keras.
# On friday, modify this to work with our data.

import numpy as np

labels=np.load('labels.np.npy')
clips=np.load('clips.np.npy')
img_rows, img_cols, img_depth = 480, 640, 5
print np.shape(clips)

clips=np.transpose(clips,(0,1,4,2,3))

print np.shape(clips)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Convolution1D

from keras.layers.convolutional_recurrent import ConvLSTM2D

from keras.layers.recurrent import LSTM


from keras.utils import np_utils, generic_utils

import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

# image specification


# Training data

batch_size = 3
nb_classes = 10
nb_epoch = 7

Y_train = np_utils.to_categorical(labels)

# Pre-processing

train_set = np.asarray(clips, dtype='float32')

train_set -= np.mean(train_set)

train_set /= np.max(train_set)

# Define model

model = Sequential()

#model.add(ConvLSTM2D(nb_filter=40,nb_col=3,nb_row=3, input_shape=(3, img_rows, img_cols, 5), return_sequences=True, border_mode='same'))

#5, 3, 480, 640

model.add(ConvLSTM2D(nb_filter=5,nb_col=3,nb_row=3, input_shape=(5, 3, 480, 640), return_sequences=True, border_mode='same'))

model.add(Convolution3D(7, 3, 3, 3,  activation='relu'))

model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(nb_classes, init='normal'))

model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=["accuracy"])

# Split the data

X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_set, Y_train, test_size=0.2, random_state=4)

# Train the model

#hist=model.train_on_batch(train_set[0:1],Y_train[0:1])

print 'trained on small batch'

hist = model.fit(X_train_new, y_train_new, validation_data=(X_val_new, y_val_new),  batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True)

# hist = model.fit(train_set, Y_train, batch_size=batch_size,
#         nb_epoch=nb_epoch,validation_split=0.2, show_accuracy=True,
#           shuffle=True)


# Evaluate the model
#score = model.evaluate(X_val_new, y_val_new, batch_size=batch_size)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])

