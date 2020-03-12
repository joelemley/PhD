# from http://learnandshare645.blogspot.ie/2016/06/3d-cnn-in-keras-action-recognition.html
# Action recognition in Keras.
# On friday, modify this to work with our data.

import scipy.ndimage as ndi # for 3d augmentation class.
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

# clipsVal=np.reshape(clipsVal,(len(clipsVal),1,10, 160, 160))
# clips=np.reshape(clips,(len(clips),1,10, 160, 160))


#(484, 16, 112, 112)

clips=np.transpose(clips,(0,4,1,2,3))
clipsVal=np.transpose(clipsVal,(0,4,1,2,3))


print np.shape(clips)
print np.shape(clipsVal)


clipsVal=np.reshape(clipsVal,(len(clipsVal),3,16, 112, 112))
clips=np.reshape(clips,(len(clips),3,16, 112, 112))

#    e.g. `input_shape=(None,3, 10, 128, 128)` for 10 frames of 128x128 RGB pictures.



import matplotlib.pyplot as plt
img=clips[6][0][1]
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


# image specification

# Training data

batch_size = 220
nb_classes = 10
nb_epoch = 100

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



#val_acc: 0.2457
model.add(Convolution3D(8, 3, 3, 3,input_shape=(3,16, 112,112),  activation='relu'))

#model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(2,2,2)))
model.add(Convolution3D(16, 3, 3, 3,  activation='relu', border_mode='same'))

#model.add(Dropout(p=.5))
#model.add(BatchNormalization())
model.add(MaxPooling3D(pool_size=(2,2,2)))
model.add(Convolution3D(32, 3, 3, 3,  activation='relu', border_mode='same'))
model.add(MaxPooling3D(pool_size=(2,2,2)))
model.add(Convolution3D(64, 3, 3, 3,  activation='relu', border_mode='same'))
model.add(Convolution3D(128, 3, 3, 3,  activation='relu', border_mode='same'))
model.add(Convolution3D(256, 3, 3, 3,  activation='relu', border_mode='same'))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(10, activation='softmax'))
sgd = Adam(lr=0.001)# wow raising learning rate from 0.0001 to 0.001 increased accuracy to 0.3674 now to val_acc: 0.3766. By contrast, the 2D network fails to learn even the training set at that learning rate.



# new best: 15713/15713 [==============================] - 17s - loss: 0.0217 - acc: 0.9976 - val_loss: 8.7364 - val_acc: 0.3957 Epoch 124/200


#sgd=SGD(lr=0.0012, decay=0.00005, momentum=0.95, nesterov=False)
from keras.optimizers import Nadam

#sgd=Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)


model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"], )

# Split the data

###X_train_new, X_val_new, y_train_new, y_val_new = train_test_split(train_set, Y_train, test_size=0.2, random_state=4)



class ThreeDImageGenerator(ImageDataGenerator):
    def transform_matrix_offset_center(self,matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix
    def apply_transform(self,x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
        #print np.shape(x)

        x=np.transpose(x,(1,0,2,3))
        #print np.shape(x)
        for xi in x:
            xi = np.rollaxis(xi, channel_index, 0)
            final_affine_matrix = transform_matrix[:2, :2]
            final_offset = transform_matrix[:2, 2]
            channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                              final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in xi]
            xi = np.stack(channel_images, axis=0)
            xi = np.rollaxis(x, 0, channel_index+1)
        x = np.transpose(x, (1, 0, 2, 3))
        #print np.shape(x)
        #print 'verified'
        return x
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



datagen = ThreeDImageGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0,  # randomly shift images vertically (fraction of total height)
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


#hist = model.fit(train_set, Y_train, validation_data=(valset, Y_trainVal),  batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True)

# hist = model.fit(train_set, Y_train, batch_size=batch_size,
#         nb_epoch=nb_epoch,validation_split=0.2, show_accuracy=True,
#           shuffle=True)


# Evaluate the model
#score = model.evaluate(X_val_new, y_val_new, batch_size=batch_size)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])

