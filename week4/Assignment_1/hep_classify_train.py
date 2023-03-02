#!/usr/bin/env python
# coding: utf-8


# Import all the necessary libraries (numpy, pandas, .. etc)
# dont forget train_test_split, roc_curve, auc

###########

import os
import warnings
warnings.filterwarnings('ignore')

# Give some output name for your file with plots, eg. output.pdf
outputname = ''
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(outputname)


###############

# Read in some number of variables from the input files
# input files are   input_WZ.txt , input_ZZ.txt 
# Look at the companion file hep_classify_plot_variables.py
# for a list of variables and their names

col_names=
cols=

# Read in the two dataframes, one for WZ and one for ZZ

WZBk =
ZZBk =

# Assign target labels for WZ and ZZ, one is 0, other is 1
# This is done by adding one additional column to each
# dataframe with that specific value
WZBk['label']=0
ZZBk['label']=1

# Merge the two dataframes into one for training
data = pd.concat([WZBk,ZZBk])


# Split the label column as y, and the input variables as X
X =
y =
print(f'Shapes of data, X, y are {data.shape}, {X.shape} , {y.shape}')


# Now we normalize the input variables to all go from -1.0 to 1.0

maxValues = X.max(axis=0)
minValues = X.min(axis=0)
MaxMinusMin = X.max(axis=0) - X.min(axis=0)
normedX = 2*((X-X.min(axis=0))/(MaxMinusMin)) -1.0
X = normedX

# print the information
print("Max values")
print(maxValues)
print("Min values")
print(minValues)


# Now we split the data into a training and a testing set
X_train, X_test, y_train, y_test =
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
n_features = X_train.shape[1]
print(f'The number of input variables is {n_features}')

# Now declare your model
model = Sequential()
model.add( )
model.add( )
model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(  )

# Train the model using model.fit
history =

# Print model summary and save the model
model.summary()
model.save( ... )


# Now make various plots

# First plot  accuracy using  history object
#  plot both accuracy and val_accuracy

# Then plot loss using both loss and val_loss


# Now here is code to plot the NN score

#Setup some new dataframes  t_df is testing, v_df is training (or validation)
t_df = pd.DataFrame()
v_df = pd.DataFrame()
t_df['train_truth'] = y_train
t_df['train_prob'] = 0
v_df['test_truth'] = y_test
v_df['test_prob'] = 0

# Now we evaluate the model on the test and train data by calling the
# predict function

val_pred_proba = model.predict(X_test)
train_pred_proba = model.predict(X_train)
t_df['train_prob'] = train_pred_proba
v_df['test_prob'] = val_pred_proba

mybins = np.arange(0,1.05,0.05)

# First we make histograms to plot the testing data as points with errors
testsig = plt.hist(v_df[v_df['test_truth']==1]['test_prob'],bins=mybins)
testsige = np.sqrt(testsig[0])
testbkg = plt.hist(v_df[v_df['test_truth']==0]['test_prob'],bins=mybins)
testbkge = np.sqrt(testbkg[0])


plt.figure(figsize=(8,8))
plt.errorbar(testsig[1][1:]-0.025, testsig[0], yerr=testsige, fmt='.', color="xkcd:green",label="Test ZZ", markersize='10')
plt.errorbar(testbkg[1][1:]-0.025, testbkg[0], yerr=testbkge, fmt='.', color="xkcd:denim",label="Test WZ", markersize='10')
plt.hist(t_df[t_df['train_truth']==1]['train_prob'],bins=mybins, histtype='step', label="Train ZZ", linewidth=3, color='xkcd:greenish',density=False,log=False)
plt.hist(t_df[t_df['train_truth']==0]['train_prob'],bins=mybins, histtype='step', label="Train WZ", linewidth=3, color='xkcd:sky blue',density=False,log=False)
plt.legend(loc='upper center')
plt.xlabel('Score',fontsize=20)
plt.ylabel('Events',fontsize=20)
plt.title(f'NN Output',fontsize=20)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
plt.yticks(fontsize=12)
#plt.savefig('NNscore.png')
plt.savefig(pp,format='pdf')


 ##  Now you add code to plot the ROC curve



 ###################

 pp.close()
 print('All done')

 
