#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import os
import warnings
warnings.filterwarnings('ignore')


# Lets set up to put all output plots in one output PDF
outputname = 'projectile_classification_week3.pdf'
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(outputname)

# 
#input file has tof,height,distance,vel,angleDegrees

# First we prepare the dataframe in which we get the data
col_names = ['tof','height','distance','velocity','angleDegrees']
cols = [0,1,2,3]
# Now here I purposely dropped the last column (read in only first 4)


projdf = pd.read_csv('input/projectile_input4_100k.txt',sep=' ',index_col=None,usecols=cols,names=col_names)

# Now let us add a column to the dataframe such that the value is 1 if velocity is between 25 and 35
# and the value is 0 otherwise
projdf['highvel'] = projdf.apply(lambda row: 1.0 if row.velocity > 25.0 and row.velocity < 35.0 else 0.0, axis=1)

# Now we are going to try make the network learn the following
#  if the velocity is between 25 and 35 (i.e. the target y value is 1.0), then NN should predict 1.0
#  else (i.e. the target y value is 0.0), the NN should predict 0.0

# Split the label column as y, and the input variables as X
X = projdf[['tof','height','distance']].values
y = projdf[['highvel']].values
print(f'Shapes of data, X, y are {projdf.shape}, {X.shape} , {y.shape}')

n_features = X.shape[1]
print(f'The number of input variables is {n_features}')

# We will now split the data into two parts, a "training" and a "testing" part.
# The idea is to use the training set to train the network, and as it trains,
# use the testing part to judge how the training is going.

# We shall split the dataset into training and testing in equal measure (50% each)
# but we can do any other combination as well.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.5)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Now we declare a neural network with 2 hidden layers
# first hidden layer has 8 neurons, and takes n_features number of inputs
# second hidden layer has 4 neurons
# output layer has 1 neuron  BUT with a sigmoid activation function
# we choose sigmoid here, since we would like the output to be interpreted as a 'probability'
# Note that our desired output is 0 or 1 ... in practice the output will be
# some value between 0 and 1. 
model = Sequential()
model.add(Dense(8, activation='relu', kernel_initializer='he_normal', input_dim=n_features))
model.add(Dense(4, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='sigmoid'))

#compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# We chose a different loss function here, as well as added 'accuracy' to the things
# the history object should retain during training.

#Now we train the model
history = model.fit(X_train,y_train,epochs=200,batch_size=100,validation_data=(X_test,y_test),verbose=0)
print(history.history.keys())

#Now we print the model summary to screen and save the model file
print('The NN architecture is')
modelname = 'my_model_disc.h5'
model.summary()
model.save(modelname)

# Let us start by making plots of the loss and accuracy as a function of epochs
# We shall do this for the usual loss and accuracy (which is evaluated on the training dataset)
# and also val_loss and val_accuracy which are evaluated for the testing dataset


plt.figure(figsize=(7,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0.001, 10])
plt.yscale('log')
plt.legend(loc='upper right')
#plt.savefig('loss_v_epoch_disc.png')
plt.savefig(pp,format='pdf')


plt.figure(figsize=(7,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.0, 1.5])
plt.legend(loc='upper right')
#plt.savefig('acc_v_epoch_disc.png')
plt.savefig(pp,format='pdf')


# We will also plot the output of the neural network, that is the score.
# We plot the score separately for 'signal' and 'background'
# here signal is the target y of 1.0, and background is the target y of 0.0
# We will plot this output separately for testing and training datasets
# Thus there are 4 curves plotted, testing/training for signal/background

#Setup some new dataframes  t_df is training, v_df is testing (or validation)
t_df = pd.DataFrame()
v_df = pd.DataFrame()
t_df['train_truth'] = y_train.ravel()
t_df['train_prob'] = 0
v_df['test_truth'] = y_test.ravel()
v_df['test_prob'] = 0

# Now we evaluate the model on the test and train data by calling the predict function
train_pred_proba = model.predict(X_train)
val_pred_proba = model.predict(X_test)

t_df['train_prob'] = train_pred_proba
v_df['test_prob'] = val_pred_proba


# Okay so now we have the two dataframes ready.
# t_df has two columns for training data  (train_truth and train_prob)
# v_df has two columns for testing data  (train_truth and train_prob)


# Now we plot the NN output
mybins = np.arange(0,1.05,0.05)


plt.figure(figsize=(7,5))
plt.hist(v_df[v_df['test_truth']==1]['test_prob'],bins=mybins,histtype='step', label="Test Signal", linewidth=3, color='xkcd:green',density=False,log=False)
plt.hist(v_df[v_df['test_truth']==0]['test_prob'],bins=mybins,histtype='step', label="Test Background", linewidth=3, color='xkcd:denim',density=False,log=False)
plt.hist(t_df[t_df['train_truth']==1]['train_prob'],bins=mybins,histtype='step',label="Train Signal", linewidth=3, color='xkcd:greenish',density=False,log=False)
plt.hist(t_df[t_df['train_truth']==0]['train_prob'],bins=mybins,histtype='step',label="Train Background", linewidth=3, color='xkcd:sky blue',density=False,log=False)
plt.legend(loc='upper center')
plt.xlabel('Score',fontsize=15)
plt.ylabel('Examples',fontsize=15)
plt.title(f'NN Output',fontsize=20)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=10)
plt.yticks(fontsize=10)
#plt.savefig('NNscore.png')
plt.savefig(pp,format='pdf')


########

# Now we get the ROC curve, first for testing data
fpr, tpr, _ = roc_curve(y_test,val_pred_proba)
auc_score = auc(fpr,tpr)
# Now the ROC curve for training data
fpr1, tpr1, _ = roc_curve(y_train,train_pred_proba)
auc_score1 = auc(fpr1,tpr1)

# We plot the ROC curves together each other to assess the performance

plt.figure(figsize=(7,5))
plt.plot(fpr,tpr,color='xkcd:denim blue', label='Testing ROC (AUC = %0.4f)' % auc_score)
plt.plot(fpr1,tpr1,color='xkcd:sky blue', label='Training ROC (AUC = %0.4f)' % auc_score1)
plt.legend(loc='lower right')
plt.title(f'ROC Curve',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.xlim(0.,1.)
plt.ylim(0.,1.)
#plt.savefig('ROC.png')
plt.savefig(pp,format='pdf')


print("All done.")
pp.close()




