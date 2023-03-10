#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pathlib
import numpy as np


import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import Sequential

#Several of these layers are new additions
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, AveragePooling2D, BatchNormalization

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import roc_curve,auc


# Now we will learn to bring in the images from some input folders
# images are in the following directories
# ./images/Training/LoPt   and   ./images/Training/HiPt
# with similar directories for the Testing.

images_dir='./images'
train_dir = os.path.join(images_dir, 'Training')
test_dir  = os.path.join(images_dir, 'Testing')
train_class1_dir = os.path.join(train_dir,'HiPt')
train_class2_dir = os.path.join(train_dir,'LoPt')
test_class1_dir = os.path.join(test_dir,'HiPt')
test_class2_dir = os.path.join(test_dir,'LoPt')
#Class1 is thus Hi, Class2 is thus Lo

print('--- Statistics on dataset ---')
print('Number of Class1 training images:', len(os.listdir(train_class1_dir)))
print('Number of Class2 training images:', len(os.listdir(train_class2_dir)))
print('---')
print('Number of Class1 test images:', len(os.listdir(test_class1_dir)))
print('Number of Class2 test images:', len(os.listdir(test_class1_dir)))
print('---')


# We can preprocess and augment the images if we choose to, by flipping them, or rotating them and so on.
# Here, we merely choose to normalize the intensities.
#imgen_train_augment = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=10, fill_mode="nearest")
imgen_train = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
imgen_test = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


# Here, be careful of the batch_size and also of the image size. The raw images are 512x512 pixels which means there are 262,144 input variables
# Using large images requires a lot of RAM and GPUs to do properly. We scale down the images to 128x128 = 16834 inputs per image
BATCH_SIZE = 32
#Rescale images to
IMG_HEIGHT = 128
IMG_WIDTH  = 128

# Now we use the flow function to feed in images batch by batch
#classes listed as background, signal -- FIRST ONE IS BKG
train_data = imgen_train.flow_from_directory(batch_size=BATCH_SIZE, shuffle=True,
                                             target_size=(IMG_HEIGHT,IMG_WIDTH),
                                             directory=train_dir,
                                             class_mode='binary',color_mode='grayscale',
                                             classes = list(['LoPt','HiPt']))

test_data = imgen_test.flow_from_directory(batch_size=BATCH_SIZE, shuffle=True,
                                           target_size=(IMG_HEIGHT,IMG_WIDTH),
                                           directory=test_dir,
                                           class_mode='binary',color_mode='grayscale',
                                           classes = list(['LoPt','HiPt']))



# We define some Callbacks , which are options we will use for the model.fit
# The model with the best validation accuracy will be saved as best_model
# Additionally we will also save the model at the end of training.
cb =  [ModelCheckpoint(filepath='best_model.h5',
                       monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')]

# Now we define the model with two convolutional layers of 32 and 64 filters each.
# The output of each is MaxPooled, and we employ BatchNormalization to reduce overfitting
# We feed the output of the conv. layers to a regular neural network.

#Define the model
model = Sequential()
model.add(Conv2D(32, (5,5), strides=(1,1), activation='relu',kernel_initializer='he_uniform', padding='SAME', input_shape=(IMG_HEIGHT,IMG_WIDTH,1)))
model.add(MaxPool2D((2,2), padding='SAME'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), strides=(1,1), activation='relu',kernel_initializer='he_uniform', padding='SAME'))
model.add(MaxPool2D((2,2), padding='SAME'))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, epochs=10, verbose=0, validation_data=test_data, callbacks=cb)
print(history.history.keys())

# Now we print the model, along with the loss and accuracy  and then we save the model
model.summary()
loss, acc = model.evaluate(test_data,verbose=0)
print('Test Accuracy: %.3f' % acc)

model.save('my_model.h5')


# Now here we make our usual set of plots of the accuracy and the loss.

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.4, 1])
plt.legend(loc='upper left')
plt.savefig('acc_v_epoch.png')
plt.close()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.001, 10])
plt.yscale('log')
plt.legend(loc='upper right')
plt.savefig('loss_v_epoch.png')
plt.close()
