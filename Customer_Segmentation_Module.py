# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 13:33:58 2022

@author: nurul
"""

#%% Module of Customer Segmentation

import numpy as np
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#%%


class cramax:
    def cramers_corrected_stat(self,confusion_matrix):
        """ calculate Cramers V statistic for categorial-categorial association.
            uses correction from Bergsma and Wicher,
            Journal of the Korean Statistical Society 42 (2013): 323-328
        """
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        phi2 = chi2/n
        r,k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    
   
class EDA:
    def visualization(self,con_col,cat_col,df):
        for con in con_col:
            plt.figure()
            sns.distplot(df[con])
            plt.show()
        for cat in cat_col:
            plt.figure()
            sns.countplot(df[cat])
            plt.show()
    def countplot_graph(self,cat_col,df):
        for i in cat_col:
            plt.figure()
            sns.countplot(df[i],hue=df['term_deposit_subscribed'])
            plt.show()
            

class ModelEvaluation():
    def plot_Acc_hist_graph(self,hist):
        plt.figure()
        plt.plot(hist.history['acc'])
        plt.plot(hist.history['val_acc'])
        plt.xlabel('epoch')
        plt.legend(['Training Acc','Validation Acc'])
        plt.show()

    def plot_loss_hist_graph(self,hist):
        plt.figure()
        plt.plot(hist.history['loss'])
        plt.plot(hist.history['val_loss'])
        plt.xlabel('epoch')
        plt.legend(['Training loss','Validation loss'])
        plt.show()
        