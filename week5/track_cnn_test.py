#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pathlib
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.metrics import roc_curve,auc

images_dir='images/'
train_dir = os.path.join(images_dir, 'Training')
test_dir  = os.path.join(images_dir, 'Testing')
train_class1_dir = os.path.join(train_dir,'HiPt')
train_class2_dir = os.path.join(train_dir,'LoPt')
test_class1_dir = os.path.join(test_dir,'HiPt')
test_class2_dir = os.path.join(test_dir,'LoPt')

print('Training dir is',train_dir)
print('Train Class1 dir is',train_class1_dir)
print('Train Class2 dir is',train_class2_dir)
print('Testing dir is',test_dir)
print('Testing Class1 dir is',test_class1_dir)
print('Testing Class2 dir is',test_class2_dir)

print('--- Statistics on dataset ---')
print('Number of Class1 training images:', len(os.listdir(train_class1_dir)))
print('Number of Class2 training images:', len(os.listdir(train_class2_dir)))
print('---')
print('Number of Class1 test images:', len(os.listdir(test_class1_dir)))
print('Number of Class2 test images:', len(os.listdir(test_class1_dir)))
print('---')

imgen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

BATCH_SIZE=32
IMG_HEIGHT = 128
IMG_WIDTH  = 128


mymodel = tf.keras.models.load_model('best_model.h5')
mymodel.load_weights('my_model.h5')


train_data_class1 = imgen.flow_from_directory(batch_size=BATCH_SIZE, shuffle=False,
                                              target_size=(IMG_HEIGHT,IMG_WIDTH),
                                              directory=train_dir, class_mode=None,
                                              classes=list(['HiPt']),color_mode='grayscale')
                                             
train_data_class2 = imgen.flow_from_directory(batch_size=BATCH_SIZE, shuffle=False,
                                              target_size=(IMG_HEIGHT,IMG_WIDTH),
                                              directory=train_dir, class_mode=None,
                                              classes=list(['LoPt']),color_mode='grayscale')

test_data_class1 = imgen.flow_from_directory(batch_size=BATCH_SIZE, shuffle=False,
                                             target_size=(IMG_HEIGHT,IMG_WIDTH),
                                             directory=test_dir,class_mode=None,
                                             classes=(['HiPt']),color_mode='grayscale')
                                             
test_data_class2 = imgen.flow_from_directory(batch_size=BATCH_SIZE, shuffle=False,
                                             target_size=(IMG_HEIGHT,IMG_WIDTH),
                                             directory=test_dir, class_mode=None,
                                             classes=(['LoPt']),color_mode='grayscale')


# This returns an array with the score and label
prob_train_Hi = mymodel.predict(train_data_class1)
prob_train_Lo = mymodel.predict(train_data_class2)

prob_test_Hi = mymodel.predict(test_data_class1)
prob_test_Lo = mymodel.predict(test_data_class2)

# These two lines combine the earlier arrays into one
# and select just first column which is the score or NN output.
prob_test = np.concatenate((prob_test_Hi,prob_test_Lo),axis=0)
prob_test = prob_test[:,0]

prob_train = np.concatenate((prob_train_Hi,prob_train_Lo),axis=0)
prob_train = prob_train[:,0]


val_Hi = np.ones(prob_test_Hi.shape)
val_Lo = np.zeros(prob_test_Lo.shape)
val_class = np.concatenate((val_Hi,val_Lo),axis=0)
val_class = val_class[:,0]
fpr, tpr, _ = roc_curve(val_class,prob_test)
auc_score = auc(fpr,tpr)


val_Hi2 = np.ones(prob_train_Hi.shape)
val_Lo2 = np.zeros(prob_train_Lo.shape)
val_class2 = np.concatenate((val_Hi2,val_Lo2),axis=0)
val_class2 = val_class2[:,0]
fpr1, tpr1, _1 = roc_curve(val_class2,prob_train)
auc_score1 = auc(fpr1,tpr1)



print('The AUC is',auc_score)
print('The train AUC is',auc_score1)



obsHi = plt.hist(prob_test_Hi[:,0],bins=np.arange(0,1.0,0.05), log=False)
obsHie = np.sqrt(obsHi[0])
obsLo = plt.hist(prob_test_Lo[:,0],bins=np.arange(0,1.0,0.05), log=False)
obsLoe = np.sqrt(obsLo[0])


plt.figure(figsize=(8,6))
plt.errorbar(obsHi[1][1:]-0.025, obsHi[0], yerr=obsHie, fmt='.', color="xkcd:green",label="Test Hi", markersize='10')
plt.errorbar(obsLo[1][1:]-0.025, obsLo[0], yerr=obsLoe, fmt='.', color="xkcd:denim",label="Test Lo", markersize='10')
plt.hist(prob_train_Hi[:,0],bins=np.arange(0,1.0,0.05), histtype='step', label="Train Hi", linewidth=3, color='xkcd:greenish',density=False,log=False)
plt.hist(prob_train_Lo[:,0],bins=np.arange(0,1.0,0.05), histtype='step', label="Train Lo", linewidth=3, color='xkcd:sky blue',density=False,log=False)
plt.legend(loc='upper center')
plt.xlabel('Score',fontsize=20)
plt.ylabel('Events',fontsize=20)
plt.title(f'NN Output',fontsize=20)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('prob_classes.png')
#plt.show()
plt.close()


plt.figure(figsize=(8,8))
plt.plot(fpr,tpr,color='xkcd:denim blue', label='ROC (AUC = %0.4f)' % auc_score)
plt.plot(fpr1,tpr1,color='xkcd:sky blue', label='Train ROC (AUC = %0.4f)' % auc_score1)
plt.legend(loc='lower right')
plt.title(f'ROC Curve',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.xlim(0.,1.)
plt.ylim(0.,1.)
plt.savefig('prob_roc.png')
plt.close()
