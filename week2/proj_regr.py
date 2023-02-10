#!/usr/bin/env python
# coding: utf-8

# First we import all the basic things we need.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

import os
import warnings
warnings.filterwarnings('ignore')


# Lets set up to put all output plots in one output PDF
outputname = 'projectile_regression_week1.pdf'
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(outputname)

# input file has tof,height,distance,vel,angleDegrees
# from this, we decide to read in  tof,height,distance,vel

# First we prepare the dataframe in which we get the data
col_names = ['tof','height','distance','velocity','angleDegrees']
cols = [0,1,2,3]
# Now here I purposely dropped the last column (read in only first 4)

projdf = pd.read_csv('projectile_input_200.txt',sep=' ',index_col=None,usecols=cols,names=col_names)


# ==================     VISUALIZE THE INPUTS     =================================
#Let us visualize the input by plotting some quantities
fig, ax = plt.subplots(2,2,figsize=(10,10))
ax = ax.flatten()

##Legend size may look bad, so let's override it
plt.rcParams['legend.fontsize']=15

ax[0].hist(projdf['tof'], bins=20, lw=2, alpha=0.5,label='ToF')
ax[0].set_title('Time of Flight')
ax[0].legend(loc='upper right')

ax[1].hist(projdf['height'], bins=20, lw=2, alpha=0.5,label='Max Height')
ax[1].set_title('Maximum Height')
ax[1].legend(loc='upper right')

ax[2].hist(projdf['distance'], bins=20, lw=2, alpha=0.5,label='Max Distance')
ax[2].set_title('Maximum Distance')
ax[2].legend(loc='upper right')

ax[3].hist(projdf['velocity'], bins=20, lw=3, alpha=0.5,label='Velocity')
ax[3].set_title('Initial Velocity')
ax[3].legend(loc='upper right')
plt.savefig(pp,format='pdf')
# ==========================================================================


# ========    CONTINUE WITH THE TRAINING ====================================

# Save the label column as y, and the input variables as X
#(X and y are numpy arrays)

X = projdf[['tof','height','distance']].values
y = projdf[['velocity']].values

print(f'Shapes of data, X, y are {projdf.shape}, {X.shape} , {y.shape}')

n_features = X.shape[1]
print(f'The number of input variables is {n_features}')


# Now we declare a neural network with 2 hidden layers
#
# first hidden layer has 16 neurons, and takes n_features number of inputs
# second hidden layer has 8 neurons
# output layer has 1 neuron
#
# We have initialized weights using option 'he_normal' and
#  we have using the ReLU activation function for all neurons.

model = Sequential()
model.add(Dense(4, activation='relu', kernel_initializer='he_normal', input_dim=n_features))
model.add(Dense(2, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='relu'))

#compile the model, by choosing a learning rate and a loss function
model.compile(optimizer='adam', loss='MeanSquaredError')


# Now we train the model
history = model.fit(X,y,epochs=10,batch_size=10,verbose=1)
print(history.history.keys())

#Now we print the model summary to screen and save the trained model to a file
modelname = 'proj_regr_model.h5'
print('The NN architecture is')
model.summary()
model.save(modelname)


# ======================================================================================


# ============== NOW VISUALIZE THE TRAINING PROCESS  ===================================
plt.figure(figsize=(10,10))
plt.plot(history.history['loss'], label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.ylim([0.001, 10])
plt.yscale('log')
plt.legend(loc='upper right')
plt.savefig(pp,format='pdf')


# ======================================================================================


# ============== EVALUATING THE MODEL  =================================================

# Now we evaluate the model using our test file.
# Let us begin by reading in the test file, and separating out the first three variables and the output

projdf_test = pd.read_csv('projectile_input_test_10.txt',sep=' ',index_col=None,usecols=cols,names=col_names)


# Then separate the variables and the result columns
X_test = projdf_test[['tof','height','distance']].values
y_true = projdf_test[['velocity']].values

# Make the prediction
pred_y = model.predict(X_test)

#At this point,  y_true has the true answers  and pred_y  has the predicted answers


# Arrange them back in a nice dataframe

results = pd.DataFrame()
results['y_true'] = y_true.ravel()
results['y_pred'] = pred_y


# Let us calculate a quick figure of merit for our sake
results['diffsquare'] = results.apply(lambda row: np.square(row.y_true-row.y_pred) , axis=1 )

#Now let us print the dataframe
print(results)

TotalDiff = results['diffsquare'].sum() / results.shape[0]
print(f"The total difference between expectation and prediction, the MSE = {TotalDiff:.4f}")

# Now we close the output file
pp.close()
