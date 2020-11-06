import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, VarianceThreshold, mutual_info_classif

"""
    Data wrapper class
    ...
    Attributes
    ----------
    X : Dataframe
        df of patients data; columns are genes, rows are patients
    X_deseq : Dataframe
        df of patient data used to run deseq2; rows are genes, columns are patient data
    y: numpy array 
        class label of patients, y[i] is label for X[i]
"""
class DataClass():
    def __init__(self, all_data, patients):
        # remove space in sample Id
        for i in range(len(patients)):
            patients['Sample ID'][i] = patients['Sample ID'][i].strip()
        
        self.patients = all_data[all_data['ID'].isin(np.array(patients['Sample ID']))]
        
        self.Ids = self.patients['ID']
        self.patient_data = patients

        # class labels 
        y = []
        for pid in self.Ids:
            if patients[patients['Sample ID'] == pid]['Compare'].values[0] == 'Yes':
                y.append(1)
            else:
                y.append(0)
        self.y = np.array(y)
        
        self.X = self.patients.reset_index().drop(['index'],axis=1)
        drop_columns = ['Unnamed: 0', 'no_feature', 'ambiguous', 'ID']
        for c in drop_columns:
            if c in self.X.columns:
                self.X = self.X.drop(columns=[c])
                
        self.X_deseq = self.X.T
        self.X_deseq = self.X_deseq.reset_index().drop(['index'],axis=1)
        self.X_deseq.columns = self.Ids        
        self.gene_names = np.array(self.X.columns)