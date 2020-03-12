# from http://learnandshare645.blogspot.ie/2016/06/3d-cnn-in-keras-action-recognition.html
# Action recognition in Keras.
# On friday, modify this to work with our data.

import keras.backend as kb

kb.set_image_dim_ordering('th')

import numpy as np

labels = np.load('labelsnotime.np.npy')
labelsVal = np.load('vallabelsnotime.np.npy')

clips = np.load('clipsnotime.np.npy')
clipsVal = np.load('valclipsnotime.np.npy')

clipsVal=np.reshape(clipsVal,(len(clipsVal),1, 60, 80))
clips=np.reshape(clips,(len(clips),1, 60, 80))

#    e.g. `input_shape=(None,3, 10, 128, 128)` for 10 frames of 128x128 RGB pictures.

import matplotlib.pyplot as plt
img=clips[10000][0]
test=plt.imshow(img,cmap='gray')
plt.show()
print np.shape(clips)

#clips=np.transpose(clips,(0,1,3,4,2))
#clipsVal=np.transpose(clipsVal,(0,1,3,4,2))
print np.shape(clips)

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Convolution1D, Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM

from keras.utils import np_utils, generic_utils

import matplotlib.pyplot as plt
import numpy as np

# image specification

# Training data

batch_size = 300 # 750 may be necessary.
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

#model = Sequential()

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


# model.add(Convolution2D(16, 3, 3,input_shape=(1, 60,80),  activation='relu'))
# model.add(Dropout(p=.5))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Convolution2D(16, 3, 3,  activation='relu', border_mode='same'))
# model.add(Dropout(p=.5))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Convolution2D(16, 3, 3,  activation='relu', border_mode='same'))
# model.add(Dropout(p=.5))
# model.add(BatchNormalization())
# model.add(Flatten())
# model.add(Dense(100, activation='relu'))
# model.add(Dense(nb_classes, init='normal'))
# model.add(Activation('softmax'))

from keras.layers import ZeroPadding2D

model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(1, 60, 80)))
model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))

model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))

model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# uncomment for tomorows experiment and compare. Tomorow also add regularization layers. Best Accuracy so far: 0.2951 with batch size 750.
# New best accuracy val_acc: 0.3062 with batch size 300
# augmentations push this up to - val_acc: 0.3876

model.add(ZeroPadding2D((1, 1), name='pool4_zp'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1), name='conv5_1_zp'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1), name='conv5_2_zp'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = Adam(lr=0.0001)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"], )

# Split the data

###X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_set, Y_train, test_size=0.2, random_state=4)

print 'trained on small batch'

#hist = model.fit(train_set, Y_train, validation_data=(valset, Y_trainVal),  batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True)

# hist = model.fit(train_set, Y_train, batch_size=batch_size,
#         nb_epoch=nb_epoch,validation_split=0.2, show_accuracy=True,
#           shuffle=True)


# Evaluate the model
#score = model.evaluate(X_val_new, y_val_new, batch_size=batch_size)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])


# this will do preprocessing and realtime data augmentation
# datagen = ImageDataGenerator(
#     featurewise_center=False,  # set input mean to 0 over the dataset
#     samplewise_center=False,  # set each sample mean to 0
#     featurewise_std_normalization=False,  # divide inputs by std of the dataset
#     samplewise_std_normalization=False,  # divide each input by its std
#     zca_whitening=False,  # apply ZCA whitening
#     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#     horizontal_flip=True,  # randomly flip images
#     vertical_flip=False)  # randomly flip images


#ABOVE GETS: 0.3876

# datagen = ImageDataGenerator(
#     featurewise_center=False,  # set input mean to 0 over the dataset
#     samplewise_center=False,  # set each sample mean to 0
#     featurewise_std_normalization=False,  # divide inputs by std of the dataset
#     samplewise_std_normalization=False,  # divide each input by its std
#     zca_whitening=False,  # apply ZCA whitening
#     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#     height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#     horizontal_flip=False,  # randomly flip images
#     vertical_flip=False)  # randomly flip images
# # above gets val_acc: 0.4631

# Note: Adding rotation is harmful. Attempted 5 and 15 degree rotations.
# Note: Very very sensitive to the types of augmentations performed.
class ThreeDImageGenerator(ImageDataGenerator):
    def test(self,a):
        return
    def random_transform(self, x): #TODO Make this work with video data.
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = self.transform_matrix_offset_center(transform_matrix, h, w)
        x = self.apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        if self.channel_shift_range != 0:
            x = self.random_channel_shift(x, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = self.flip_axis(x, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = self.flip_axis(x, img_row_index)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(train_set)

# fit the model on the batches generated by datagen.flow()
model.fit_generator(datagen.flow(train_set, Y_train,
                                 batch_size=batch_size),
                    samples_per_epoch=train_set.shape[0],
                    nb_epoch=nb_epoch,
                    validation_data=(valset, Y_trainVal))
