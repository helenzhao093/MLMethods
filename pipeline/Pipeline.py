from sklearn import base 
from sklearn.datasets import load_diabetes, load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, ShuffleSplit
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.svm import SVR, SVC, LinearSVC, LinearSVR
from sklearn.preprocessing import normalize, MinMaxScaler, StandardScaler, KBinsDiscretizer, label_binarize
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.ensemble import BaggingClassifier
import seaborn as sns
import statsmodels.api as sm
import os
import warnings

class ClassifierRunner():
    def __init__(self, pipeline, clf, clf_name, example_indices, selector_params=None, multiclass=None, compare_classifiers=None) :
        self.pipeline = pipeline
        self.clf = clf
        self.clf_name = clf_name
        self.example_indices = example_indices
        self.selector_params = selector_params
        self.selector_param_str = self.selector_params['combine'] + str(self.selector_params['threshold'])
        self.multiclass = multiclass
        self.compare_classifiers = compare_classifiers 
    
    def fit(self, X_train_tranformed, y_train, num_classes):
        self.num_classes = num_classes
        self.num_features = len(X_train_tranformed[0])

        if hasattr(self.clf, 'reset'): # reset classifier from sklearn library
            self.clf = self.clf.reset()
        else:
            self.clf = base.clone(self.clf)
            if self.num_classes > 2:
                if self.multiclass=='ovr':
                    self.clf = OneVsRestClassifier(self.clf) 
                else:
                    self.clf = OneVsOneClassifier(self.clf)
        self.clf.fit(X_train_tranformed, y_train)
        
    def run_prediction(self, X_test_transformed, y_test):
        # get predictions
        y_pred = self.clf.predict(X_test_transformed)

        self.proba = None
        if (self.num_classes == 2 or self.multiclass == 'ovr') and self.clf.__class__.__name__ != 'SVC':
            self.proba = self.clf.predict_proba(X_test_transformed)

        # get ids of examples that were misclassified
        if self.compare_classifiers == 'mcnemar':
            for i in range(len(self.y_pred)):
                if y_pred[i] != y_test[i]:
                    self.pipeline.misclassified_map[self.clf_name].append(self.example_indices[i])

        self.write_predictions(y_pred)
                
        self.write_metrics(y_test, y_pred)

    def write_predictions(self, y_pred):
        if self.proba is not None and y_pred is not None:
            for i in range(len(y_pred)):
                self.pipeline.prediction_scores[self.selector_param_str][self.clf_name][self.example_indices[i]] = self.proba[i]
                self.pipeline.predictions[self.selector_param_str][self.clf_name][self.example_indices[i]] = y_pred[i] 

        
    def write_metrics(self, y_test, y_pred):
        average='binary' if self.num_classes > 2 else 'micro'
        auc_score = []
        if self.proba is not None:
            if average == 'binary':
                auc_score = roc_auc_score(y_test, y_pred) 
            else:
                y_bin = label_binarize(y_test, classes=range(self.num_classes))
                for i in range(self.num_classes):
                    y_temp = y_bin[:,i]
                    auc_score.append(roc_auc_score(y_temp, self.proba[:, i]))
                auc_score = str(auc_score)

        self.pipeline.results = self.pipeline.results.append({
                                'base clf' : self.clf_name,
                                'num features': self.num_features,
                                'accuracy' : accuracy_score(y_test, y_pred), 
                                'precision': precision_score(y_test, y_pred, average=average),  
                                'recall': recall_score(y_test, y_pred, average=average), 
                                'auc': auc_score if auc_score is not None else -1, 
                                'f1' : f1_score(y_test, y_pred, average=average) }, 
                                ignore_index=True)
        
        if self.selector_params is not None:
            for k in self.selector_params.keys():
                self.pipeline.results.loc[len(results)-1, k] = self.selector_params[k]
        

class DeseqEmsembleSelectorPipeline():
    def __init__(self, emsemble_selector, clfs, clf_names, combine_param_map=[], multiclass=None):
        #self.data = data_wrapper
        self.selector = emsemble_selector
        self.clfs = clfs 
        self.clf_names = clf_names
        self.combine_param_map = combine_param_map
        self.multiclass = multiclass

    def initialize_data_storage(self, X):
        self.results = pd.DataFrame(columns=["base clf", "combine", "threshold", "num features", "accuracy", "precision", "recall", "auc", "f1"])
        
        # initialize parameter strings (e.g min50)
        param_strs = []
        for combine in self.combine_param_map.keys():
            for threshold in self.combine_param_map[combine]:
                param_strs.append(combine + str(threshold))

        # initialize df of predicition score for each sample for each emsemble parameter 
        self.prediction_scores = initialize_prediction_scores_map_for_params(param_strs, self.clf_names, len(X)) if len(param_strs) > 0 else initialize_prediction_scores_map(self.clf_names, len(X))
        self.predictions = initialize_prediction_scores_map_for_params(param_strs, self.clf_names, len(X)) if len(param_strs) > 0 else initialize_prediction_scores_map(self.clf_names, len(X))

        # dict of misclassified example indices for each classifier for each emsemble parameter
        self.misclassified_map = initialize_classifier_map_for_params(param_strs, self.clf_names) if len(param_strs) > 0 else initialize_classifier_map(base_clf_names)
    
        self.selection_folds = initialize_classifier_map(param_strs) if len(param_strs) > 0 else []

        self.results_params = dict()

    def fit(self, X, y, Ids, cv_size=5, directory=None, filename=None, multiclass='ovr', compare_classifiers=None):
        self.initialize_data_storage(X)
        
        CVSplitter = StratifiedKFold(n_splits=cv_size, random_state=None, shuffle=True)
        self.num_classes = np.max(y) + 1

        for train_index, test_index in CVSplitter.split(X, y): 
            self.test_index = test_index 
            X_train_df = X.loc[train_index]
            X_test_df = X.loc[test_index]

            #self.X_train = X[train_index] #X.loc[train_index]
            #self.X_test = X[test_index]#X.loc[test_index]
            self.y_train, self.y_test = y[train_index], y[test_index]

            # run deseq2 on training data
            selected_patients = patient_data.loc[train_index]
            deseqRunner = DeseqRunner(X_train_df.T, selected_patients, "~ Compare")
            self.genenames = deseqRunner.run()

            # select only deseq genes in dataframe and normalize gene counts
            normalizer = SizeFactorNormalize(deseqRunner.geneToGeomean)
            self.X_test = normalizer.normalize(X_test_df[deseq_genes].columns, X_test_df[deseq_genes])
            self.X_train = normalizer.normalize_with_size_factors(X_train_df[deseq_genes], deseqRunner.size_factors)
        
            # run emsemble feature selection
            #if len(self.combine_param_map) > 0: 
            #    self.selector.fit(self.X_train, self.y_train, combine=False)
            #else:
            self.selector.fit(self.X_train, self.y_train)
            
            # run classifiers 
            self.run_classifiers()
            
    
    def run_classifiers(self):
        for combine in self.combine_param_map.keys():
            thresholds = self.combine_param_map[combine]
            max_threshold = max(thresholds)
            if max_threshold > 0:
                self.selector.combine_rankings(combine, max_threshold)
            for threshold in thresholds:
                selector_param_str = combine + str(threshold)
                selector_params['combine'] = combine
                selector_params['threshold'] = threshold
                
                selection_indices = self.selector.selection_indices[:threshold] if threshold > 0 else selector.selected_indices
                selected_genenames = []
                for i in selection_indices:
                    selected_genenames.append(self.genenames[i])
                
                self.selection_folds[selector_param_str].append(selected_genenames)

                X_train_tranformed = self.X_train[:, self.selector.selection_indices[:threshold]] if threshold > 0 else selector.transform(self.X_train)
                X_test_transformed = self.X_test[:, self.selector.selection_indices[:threshold]] if threshold > 0 else selector.transform(self.X_test)
                
                for i in range(len(self.clfs)):
                    clfrunner = ClassifierRunner(self, self.clfs[i], self.clf_names[i], self.test_index, self.selector_params, self.multiclass)
                    clfrunner.fit(X_train_tranformed, self.y_train, self.num_classes)
                    clfrunner.run_prediction(X_test_transformed, self.y_test)              
    
    def write_results(self, directory, folder):
        new_dir = make_dir_if_not_exist(directory, folder)
        for key_str in self.prediction_scores.keys():
            prediction_scores = self.prediction_scores[key_str]
            create_scores_csv(num_classes, y, prediction_scores, patient_ids, os.path.join(new_dir, key_str))
        
        if self.compare_classifiers=='mcnemar':
            for param in self.misclassified_map.keys():
                compare_all(self.misclassified_map[param], len(X), os.path.join(new_dir, param))
        
        results.groupby(['base clf', 'combine', 'threshold']).mean().to_csv(os.path.join(new_dir, filename + '_mean' + ".csv"))

        
                
def make_dir_if_not_exist(directory, folder):
    new_dir = None
    if directory is not None:
        new_dir = os.path.join(directory, folder)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
    return new_dir

# helpers 
def initialize_prediction_scores_map_for_params(param_strs, base_clf_names, sample_size):
    all_prediction_scores = dict()
    for param_str in param_strs: 
        all_prediction_scores[param_str] = initialize_prediction_scores_map(base_clf_names, sample_size)
    return all_prediction_scores

def initialize_prediction_scores_map(base_clf_names, sample_size):
    prediction_scores = dict()
    for name in base_clf_names:
        if name[0:3] != 'SVC':
            prediction_scores[name] = [[0,0]] * sample_size
    return prediction_scores

def initialize_classifier_map(base_clf_names):
    classifer_map = dict()
    for name in base_clf_names:
        classifer_map[name] = []
    return classifer_map

def initialize_classifier_map_for_params(param_strs, base_clf_names):
    param_to_classifiers_map = dict()
    for param in param_strs:
        param_to_classifiers_map[param] = initialize_classifier_map(base_clf_names)
    return param_to_classifiers_map 