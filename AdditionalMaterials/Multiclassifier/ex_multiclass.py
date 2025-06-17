#!/usr/bin/env python
# coding: utf-8

#
# Run this code as follows
#
# ex_multiclass.py <MODELNAME> <PDFNAME>
#
# where <MODELNAME> is the name of output model name (say my_model.h5)
# and <PDFNAME> is the name of the output PDF file with plots (say output.pdf)
#

#Import the necessary packages
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve,auc
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense
#from tensorflow.keras.utils import np_utils

import os
import sys
import warnings
warnings.filterwarnings('ignore')

modelname = sys.argv[1]
outputname = sys.argv[2]

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(outputname)

#This is the full list of input variables in the text file
#col_names=['Pt0',' Pt1',' Pt2','Pt3','NBJet','MET','LQSum','LepMt0','LepMt1','LepMt2','LepMt3','HT','LLBestMOSSF','MaxDphi_LMet','MinDphi_LMet','MaxDphi_LL','MinDphi_LL','LMOSOF','TBestMOSOF','TBestMOSSF','LLPairPt','LLSSSFMass','LLSSOFMass','TMSSOF','Mt0','Mt1','Mt2','LDrmin','LMass','L3BelowZ','L3OnZ','L3AboveZ','L3SS','L4DoubleOnZ','L4SingleOnZ','L4OffZ','L2T1BelowZ','L2T1OnZ','L2T1AboveZ','L2T1SS','L1T2OSLowMT','L1T2OSHighMT','L1T2SS','Rare3L1TOnZ','Rare3L1TOffZ','Rare2L2TOnZ','Rare2L2TOffZ','Rare1L3TOffZ'];
#cols = list(range(0,48))

#We start by using a small subset
col_names=['Pt0','Pt1','Pt2','NBJet','MET','MaxDphi_LMet','MaxDphi_LL','MinDphi_LL','LLPairPt','Mt0','Mt1','Mt2']
cols = [0,1,2,4,5,13,15,16,20,24,25,26]

WZBk = pd.read_csv('inputs/T3L_WZ.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
WZBk['label']=1

ZZBk = pd.read_csv('inputs/T3L_ZZ.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
ZZBk['label']=0

TTBk = pd.read_csv('inputs/T3L_TTZ.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
TTBk['label']=2


data = pd.concat([WZBk,ZZBk,TTBk])
X, y = data.values[:,:-1], data.values[:,-1]
#Here the y values, or labels are turned from 0,1,2 into
# one hot encoded values (1,0,0),(0,1,0),(0,0,1)
ohe_y = tf.keras.utils.to_categorical(y)

maxValues = X.max(axis=0)
minValues = X.min(axis=0)
print("Max values")
print(maxValues)
print("Min values")
print(minValues)
MaxMinusMin = X.max(axis=0) - X.min(axis=0)
normedX = 2*((X-X.min(axis=0))/(MaxMinusMin)) -1.0
X = normedX


X_train, X_test, y_train, y_test = train_test_split(X,ohe_y,test_size=0.5)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
n_features = X_train.shape[1]
print(f'The number of input variables is {n_features}')


model = Sequential()
model.add(Dense(256, activation='relu', kernel_initializer='he_normal', input_dim=n_features))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(16, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax'))

#compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=20,batch_size=512,validation_data=(X_test,y_test),verbose=0)
                 
#Now we print the model summary to screen and save the model file
print('The NN architecture is')
model.summary()
model.save(modelname)


# Thats it. Now the rest of this file is just making various plots


# Let us start by making plots of the accuracy and loss as a function of epochs
# this tells us how the training went.
plt.figure(figsize=(7,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label = 'Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.4, 1])
plt.legend(loc='upper left')
#plt.savefig('acc_v_epoch.png')
plt.savefig(pp, format='pdf')
plt.close()

plt.figure(figsize=(7,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label = 'Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.001, 10])
plt.yscale('log')
plt.legend(loc='upper right')
#plt.savefig('loss_v_epoch.png')
plt.savefig(pp, format='pdf')
plt.close()

pp.close()
print(f'All done. Output is saved as {outputname}')
