#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import warnings
warnings.filterwarnings('ignore')

outputname = 'hep_classify_output_var.pdf'
from matplotlib.backends.backend_pdf import PdfPages
pp = PdfPages(outputname)

col_names=['Pt0','Pt1','Pt2','NBJet','Met','MaxDphi_LMet','MaxDphi_LL','MinDphi_LL','LLPairPt','Mt0','Mt1','Mt2']
cols = list(range(0,12))

WZBk = pd.read_csv('input_WZ.txt',sep=' ',index_col=None, usecols=cols,names=col_names)
ZZBk = pd.read_csv('input_ZZ.txt',sep=' ',index_col=None, usecols=cols,names=col_names)




def plotVars(mybins,plotnames):
    for pname in plotnames:
        plt.figure(figsize=(8,8))
        plt.hist(WZBk[pname],bins=mybins,histtype='step',label="WZ",linewidth=3, color='blue',density=False,log=False)
        plt.hist(ZZBk[pname],bins=mybins,histtype='step',label="ZZ",linewidth=3, color='red',density=False,log=False)
        plt.legend(loc='upper center')
        plt.xlabel(pname,fontsize=20)
        plt.ylabel('Entries',fontsize=20)
        plt.title(pname,fontsize=20)
        plt.savefig(pp,format='pdf')
        plt.close()


plotnames=['Pt0','Pt1','Pt2','Met','LLPairPt','Mt0','Mt1','Mt2']
mybins = np.arange(0,1000,20)
plotVars(mybins,plotnames)

plotnames=['MaxDphi_LMet','MaxDphi_LL','MinDphi_LL']
mybins = np.arange(0,3.2,0.1)
plotVars(mybins,plotnames)

plotnames=['NBJet']
mybins = np.arange(0,10,1)
plotVars(mybins,plotnames)

pp.close()
