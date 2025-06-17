
Multiclassification task
=========================
- ex_multiclass.py
In this example, we show a multiclassifier for three processes - the network has three output
neurons, each giving a probability for the input to be of a particular class.
The inputs are the two from example above, and inputs/T3L_TTZ.txt.
The outputs are the trained model file (h5 file) and a pdf with output plots.

- ex_multiclass_test.py
This shows how to read in a trained model to make predictions.
Here one has to give the trained model as an argument, and one has to edit the max/min val arrays
based on the output given by the ex_multiclass.py script.
This script has two functions:
 a) take one dataframe and plot all three output neuron scores
 b) take one output neuron, and plot output of all three processes on this neuron.

Input details for the classification task:

- WZ: T3L_2017_WZ_100K.txt
- ZZ: T3L_2017_ZZ_100K.txt
- TTZ: T3L_2017_TTZ_100K.txt

Contact Arnab/Sourabh to get the files. Variables are:

Pt0 ,  Pt1 ,  Pt2 , Pt3 , NBJet , MET , LQSum , LepMt0 , LepMt1 , LepMt2 , LepMt3 , HT , LLBestMOSSF ,
MaxDphi_LMet , MinDphi_LMet , MaxDphi_LL , MinDphi_LL , LMOSOF , TBestMOSOF , TBestMOSSF , LLPairPt ,
LLSSSFMass , LLSSOFMass , TMSSOF , Mt0 , Mt1 , Mt2 , LDrmin , LMass ,
L3BelowZ , L3OnZ , L3AboveZ , L3SS , L4DoubleOnZ , L4SingleOnZ , L4OffZ , L2T1BelowZ , L2T1OnZ , L2T1AboveZ , L2T1SS ,
L1T2OSLowMT , L1T2OSHighMT , L1T2SS , Rare3L1TOnZ , Rare3L1TOffZ , Rare2L2TOnZ , Rare2L2TOffZ , Rare1L3TOffZ'
