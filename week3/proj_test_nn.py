#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc

import os
import warnings
warnings.filterwarnings('ignore')


# Lets set up to put all output plots in one output PDF
outputname = 'projectile_classification_test_week3.pdf'
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(outputname)


# First we prepare the testing dataframe

col_names = ['tof','height','distance','velocity','angleDegrees']
cols = [0,1,2,3]

projdf = pd.read_csv('input/projectile_input4_test_20k.txt',sep=' ',index_col=None,usecols=cols,names=col_names)
projdf['highvel'] = projdf.apply(lambda row: 1.0 if row.velocity > 25.0 and row.velocity < 35.0 else 0.0, axis=1)

# Now split into the input vector and the targeted true output
X = projdf[['tof','height','distance']].values
y_true = projdf[['highvel']].values
print(f'Shapes of data, X, y_true are {projdf.shape}, {X.shape} , {y_true.shape}')

# Now we load the model we have trained already and its weights
modelname = 'my_model_disc.h5'
mymodel = tf.keras.models.load_model(modelname)
mymodel.load_weights(modelname)


# Let us now predict based on the model
# Now we evaluate the model on the test and train data by calling the predict function
pred_proba = mymodel.predict(X)


#Put it all together in a dataframe
results = pd.DataFrame()
results['true_y']  = y_true.ravel()
results['pred_y'] = pred_proba


#Get the ROC curve
fpr, tpr, _ = roc_curve(y_true,pred_proba)
auc_score = auc(fpr,tpr)


#Plot score, and then the ROC curve


mybins = np.arange(0,1.05,0.05)

plt.figure(figsize=(7,5))
plt.hist(results[results['true_y']==1]['pred_y'],bins=mybins,histtype='step', label="Test Signal", linewidth=3, color='xkcd:denim',density=False,log=False)
plt.hist(results[results['true_y']==0]['pred_y'],bins=mybins,histtype='step', label="Test Background", linewidth=3, color='xkcd:sky blue',density=False,log=False)
plt.legend(loc='upper center')
plt.xlabel('Score',fontsize=15)
plt.ylabel('Examples',fontsize=15)
plt.title(f'NN Output for Test',fontsize=20)
plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=10)
plt.yticks(fontsize=10)
#plt.savefig('NNscore.png')
plt.savefig(pp,format='pdf')


plt.figure(figsize=(7,5))
plt.plot(fpr,tpr,color='xkcd:denim blue', label='Testing ROC (AUC = %0.4f)' % auc_score)
plt.legend(loc='lower right')
plt.title(f'ROC Curve for Test',fontsize=20)
plt.xlabel('False Positive Rate',fontsize=20)
plt.ylabel('True Positive Rate',fontsize=20)
plt.xlim(0.,1.)
plt.ylim(0.,1.)
#plt.savefig('ROC.png')
plt.savefig(pp,format='pdf')


print("All done.")
pp.close()





# Now let us evaluate the model using our test file

## First we read in the test file.
#projdf_test = pd.read_csv('projectile_regression/projectile_input_test_100.txt',sep=' ',index_col=None,usecols=cols,names=col_names)
#projdf_test['highvel'] = projdf_test.apply(lambda row: 1.0 if row.velocity > 15.0 else 0.0, axis=1)
## Then separate the variables and the result columns
#X_test = projdf_test[['tof','height','distance']].values
#y_test = projdf_test[['highvel']].values


## Make the prediction
#pred_y = model.predict(X_test)

## Arrange them back in a nice dataframe

#results = pd.DataFrame()
#results['y_true'] = y_test.ravel()
#results['y_pred'] = pred_y

##Now let us print the dataframe
#print(results)


