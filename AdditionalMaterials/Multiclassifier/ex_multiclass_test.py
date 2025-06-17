#!/usr/bin/env python
# coding: utf-8

#
# Run this code as follows
#
# ex_multiclass_test.py <MODELNAME> <PDFNAME>
#
# where <MODELNAME> is the name of input model name (say my_model.h5)
# and <PDFNAME> is the name of the output PDF file with plots (say output.pdf)
#

#Import the necessary packages
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import sys
import warnings
warnings.filterwarnings('ignore')

modelname = sys.argv[1]
outputname = sys.argv[2]

from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(outputname)

#First we put in the same columns here that we have in the mynn.py
col_names=['Pt0','Pt1','Pt2','NBJet','MET','MaxDphi_LMet','MaxDphi_LL','MinDphi_LL','LLPairPt','Mt0','Mt1','Mt2']
cols = [0,1,2,4,5,13,15,16,20,24,25,26]

WZBk = pd.read_csv('inputs/T3L_WZ.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
ZZBk = pd.read_csv('inputs/T3L_ZZ.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
TTBk = pd.read_csv('inputs/T3L_TTZ.txt',sep=' ',index_col=None, usecols=cols,names=col_names)

alldfs = [WZBk,ZZBk,TTBk]
alldfnames = ['WZ','ZZ','TTZ']

#=====================================
#Now we must declare arrays to keep the max and min values that we got from ex_multiclass.py
maxVals = np.array([[1054.561646,  847.422913,  298.373138,    5.,       1144.37146,     3.141569,
                     3.141525,    2.089597,  953.577454, 1937.036621, 1835.078247,  845.757935]])
minVals = np.array([[ 2.9002983e+01,  1.0053716e+01,  1.0002108e+01,  0.0000000e+00,
                      3.8575400e-01,  8.9869000e-02,  4.0797000e-02,  0.0000000e+00,
                      -1.0000000e+00,  1.3890000e-03,  4.6250000e-03,  4.2250000e-03]])


mymodel = tf.keras.models.load_model(modelname)
mymodel.load_weights(modelname)



mybins = np.arange(0,1.05,0.05)

#This function takes a dataframe and plots all three scores for it
def processdf(df,dfname):
    normeddf = 2*( (df-minVals)/(maxVals-minVals) ) -1.0
    df = normeddf
    nnscore = mymodel.predict(df)
    plt.figure(figsize=(8,6))
    plt.hist(nnscore[:,0],bins=mybins,histtype='step',label='ZZneuron',linewidth=3,color='xkcd:sky blue',density=False,log=True)
    plt.hist(nnscore[:,1],bins=mybins,histtype='step',label='WZneuron',linewidth=3,color='xkcd:red',density=False,log=True)
    plt.hist(nnscore[:,2],bins=mybins,histtype='step',label='TTZneuron',linewidth=3,color='xkcd:green',density=False,log=True)
    plt.legend(loc='upper center')
    plt.xlabel('Score',fontsize=20)
    plt.ylabel('Events',fontsize=20)
    plt.title(f'NN Output for {dfname}',fontsize=20)
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
    plt.yticks(fontsize=12)
    #Save to file instead of individual image
    #plt.savefig('score'+dfname+'.png')
    plt.savefig(pp, format='pdf')
    plt.close()

#This function takes a list of dataframes and plots a specific score for it
mycols=['xkcd:sky blue','xkcd:red','xkcd:green']
labelnames = ['ZZneuron','WZneuron','TTZneuron']
def processdflist(dflist,dfnames,proclabel):

    plt.figure(figsize=(8,6))
    for index, df in enumerate(dflist):
        normeddf = 2*( (df-minVals)/(maxVals-minVals) ) -1.0
        df = normeddf
        nnscore = mymodel.predict(df)
        plt.hist(nnscore[:,proclabel],bins=mybins,histtype='step',label=dfnames[index],
                 linewidth=3,color=mycols[index],density=False,log=True)
    plt.legend(loc='upper center')
    plt.xlabel('Score',fontsize=20)
    plt.ylabel('Events',fontsize=20)
    plt.title(f'{labelnames[proclabel]} NN Output',fontsize=20)
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],fontsize=12)
    plt.yticks(fontsize=12)
    #plt.savefig('score'+labelnames[proclabel]+'.png')
    plt.savefig(pp, format='pdf')    
    plt.close()
    
    
processdf(WZBk,'procWZ')
processdf(ZZBk,'procZZ')
processdf(TTBk,'procTTZ')
processdflist(alldfs,alldfnames,0)
processdflist(alldfs,alldfnames,1)
processdflist(alldfs,alldfnames,2)


pp.close()
print(f'All done. Output is saved as {outputname}')
