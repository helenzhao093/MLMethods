import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, VarianceThreshold, mutual_info_classif
import matplotlib.pyplot as plt

class DataClass():
    def __init__(self, all_data, patients, deseq_genes=None, multiclass=False, variance_threshold=0.0):
        self.patients = all_data[all_data['ID'].isin(np.array(patients['Sample ID']))]
        self.Ids = self.patients['ID']

        # class labels 
        y = []
        self.groups = []
        if not multiclass:
            for pid in self.Ids:
                if patients[patients['Sample ID'] == pid]['Compare'].values[0] == 'Yes':
                    y.append(1)
                else:
                    y.append(0)

                if 'Group' in patients.columns:
                    self.groups.append(patients[patients['Sample ID'] == pid]['Group'].values[0])
        else:
            for pid in self.Ids:
                y.append(patients[patients['Sample ID'] == pid]['Compare'].values[0])

            if 'Group' in patients.columns:
                self.groups.append(patients[patients['Sample ID'] == pid]['Group'].values[0])
        
        self.y = np.array(y)
        
        # set deseq genes columns
        if deseq_genes is not None:
            self.deseq_genes_names = deseq_genes['name']
            drop_columns = ['Unnamed: 0', 'no_feature', 'ambiguous', 'ID']
            for c in drop_columns:
                if c in self.deseq_genes_names:
                    self.deseq_genes_names.drop(c)
            self.X_deseq_org = self.patients[self.deseq_genes_names]
        
            scaler = StandardScaler()
            scaler.fit(self.X_deseq_org)
            self.X_deseq = scaler.transform(self.X_deseq_org)

        # set X 
        self.X = self.patients
        for c in drop_columns:
            if c in self.X.columns:
                self.X = self.X.drop(columns=[c])
        
        #drop_columns = []
        #for c in self.X.columns:
        #    if self.X[c].sum() == 0:
        #        drop_columns.append(c)
        #self.X_org = self.X.drop(columns=drop_columns)

        self.gene_names = np.array(self.X.columns)

        # remove low variance features 
        selector = VarianceThreshold(threshold=variance_threshold)
        self.X_org = selector.fit_transform(self.X)
        self.gene_names = selector.transform([self.gene_names])[0]
        
        scaler = StandardScaler()
        scaler.fit(self.X_org)
        self.X = scaler.transform(self.X_org)

    def plot_chi_sq(self):
        chi_sq = chi2(self.X, self.y)[0]
        sorted_chi_sq = -np.sort(-chi_sq)
        self.plot_score(sorted_chi_sq)

    def plot_mutual_info(self):
        mi = mutual_info_classif(self.X, self.y)
        sorted_mi = -np.sort(-mi)
        self.plot_score(sorted_mi)

    def plot_score(self, score):
        plt.figure(figsize=(16,8))
        plt.plot(score)
        plt.xlabel('# of feature')
        plt.ylabel('score')
        plt.show()

    def transform_chi_sq_threshold(self, threshold):
        chi_sq = chi2(self.X, self.y)[0]
        self.transform_score(chi_sq, threshold)

    def transform_MI_threshold(self, threshold):
        mi = mutual_info_classif(self.X, self.y)
        self.transform_score(mi, threshold)

    def transform_score(self, scores, threshold):
        selected_indices = []
        for i, score in enumerate(scores):
            if score > threshold:
                selected_indices.append(i)
        self.X_transformed = self.X[:,selected_indices]
        self.genes_names_transformed = self.gene_names[selected_indices]