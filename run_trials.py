from .emsemble.HomogeneousEmsemble import HomogeneousEmsemble
from .emsemble.Emsemble import Emsemble
from .feature_selection.EmsembleFS import EmsembleFS
from .feature_selection.RFRFE import RFRFE
from .feature_selection.MSVMRFE import MSVMRFE
from .feature_selection.mRMR import mRMR
from .feature_selection.Relief import Relief
from .feature_selection.AGA import AGA
from .classification.ThresholdClassifier import ThresholdClassifier
from .classification.BaselineClassifiers import BaselineClassifiers

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

def make_dir_if_not_exist(directory, filename):
    new_dir = None
    if directory is not None:
        new_dir = os.path.join(directory, filename)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
    return new_dir

def add_row(results, y_test, y_pred, base_clf, num_features, num_classes, prediction_proba):
    average='binary'
    if num_classes > 2:
        average='micro'
    auc_score = None
    if prediction_proba is not None:
        if average == 'binary':
            auc_score = roc_auc_score(y_test, y_pred) 
        else:
            auc_score = []
            classes = range(num_classes)
            y_bin = label_binarize(y_test, classes=classes)
            for i in range(num_classes):
                y_temp = y_bin[:,i]
                auc_score.append(roc_auc_score(y_temp, prediction_proba[:, i]))
            auc_score = str(auc_score)

    results = results.append({'base clf' : base_clf,
                              'num features': num_features,
                              'accuracy' : accuracy_score(y_test, y_pred), 
                              'precision': precision_score(y_test, y_pred, average=average),  
                              'recall': recall_score(y_test, y_pred, average=average), 
                              'auc': auc_score if auc_score is not None else -1, 
                              'f1' : f1_score(y_test, y_pred, average=average)}, 
                              ignore_index=True)
    return results

def add_classifier(results, all_pred, all_pred_scores, 
                   X_train_tranformed, y_train, X_test_transformed, y_test, 
                   clf_temp, clf_name, 
                   results_params, num_classes, multiclass,
                   example_indices, compare_classifiers, misclassified):

    clf = None
    if hasattr(clf_temp, 'reset'): # emsembles
        clf = clf_temp.reset()
    else:
        clf = base.clone(clf_temp)
        if num_classes > 2:
            if multiclass=='ovr':
                clf = OneVsRestClassifier(clf)
            else:
                clf = OneVsOneClassifier(clf)
    
    # get predictions
    clf.fit(X_train_tranformed, y_train)
    y_pred = clf.predict(X_test_transformed)
    all_pred[clf_name] = y_pred
    
    # get ids of examples that were misclassified
    if compare_classifiers == 'mcnemar':
        for i in range(len(y_pred)):
            if y_pred[i] != y_test[i]:
                misclassified[clf_name].append(example_indices[i])

    # get prediction probabilities 
    proba = None
    if (num_classes == 2 or multiclass == 'ovr') and clf.__class__.__name__ != 'SVC':
        proba = clf.predict_proba(X_test_transformed)
        all_pred_scores[clf_name] = proba

    # add performance results to table
    results = add_row(results, y_test, y_pred, clf_name, len(X_train_tranformed[0]), num_classes, proba)
    if results_params is not None:
        for k in results_params.keys():
            results.loc[len(results)-1, k] = results_params[k]
    return results

def run_classifiers(results, base_clfs, base_clf_names,
                    X_train_tranformed, y_train, X_test_transformed, y_test, 
                    num_classes, multiclass, results_params,
                    example_indices, compare_classifiers, misclassified):
    all_pred_scores = dict()
    all_pred = dict()

     # create homogeneous emsemble
    for i in range(len(base_clfs)):
        results = add_classifier(results, all_pred, all_pred_scores, 
                                 X_train_tranformed, y_train, X_test_transformed, y_test, 
                                 base_clfs[i] , base_clf_names[i], 
                                 results_params, num_classes, multiclass,
                                 example_indices, compare_classifiers, misclassified)
    return results, all_pred, all_pred_scores


def create_scores_csv(num_classes, y, prediction_scores, ids, directory):
    if num_classes == 2:
        create_scores_csv_binary(y, prediction_scores, ids, directory)
    else:
        create_scores_csv_multiclass(y, prediction_scores, ids, directory)

def create_scores_csv_multiclass(y, prediction_scores, ids, directory):
    num_classes = np.max(y)
    if not os.path.exists(directory):
        os.mkdir(directory)
    for clf in prediction_scores.keys():
        scores = np.array(prediction_scores[clf])
        correct = []
        wrong = []
        for i in range(len(scores)): 
            true_class = y[i] * 1.0
            predicted_class = np.argmax(scores[i]) * 1.0
            if true_class == predicted_class:
                # TP 
                correct.append(np.concatenate(( [ids[i], true_class, predicted_class] , scores[i])))
            else:
                wrong.append(np.concatenate(( [ids[i], true_class, predicted_class] , scores[i])))
            
        names = ['correct', 'wrong'] 
        data_columns = ['Id', 'True Class', 'Predicted Class']
        for c in range(num_classes + 1):
            data_columns.append('score class ' + str(c))
        for i, data in enumerate([correct, wrong]):
            data = sorted(data,key=lambda x: x[1])
            df = pd.DataFrame(data, columns=data_columns)
            df.to_csv(os.path.join(directory, clf + "_" + names[i] + ".csv"), index=False)
            
# find TPs with high prediction score
# FNs with low score for prediction of 1 
def create_scores_csv_binary(y, prediction_scores, ids, directory):
    num_classes = np.max(y)
    if not os.path.exists(directory):
        os.mkdir(directory)

    for clf in prediction_scores.keys():
        scores = np.array(prediction_scores[clf])
        neg_score = scores[:,0]
        pos_score = scores[:,1]
        TP, FP, TN, FN = [], [], [], []
        for i in range(len(scores)): 
            if y[i] == 1: # true class
                if pos_score[i] > neg_score[i]: #TP
                    TP.append([ ids[i], neg_score[i], pos_score[i]])
                else: #FN
                    FN.append([ ids[i], neg_score[i], pos_score[i]])
            else:
                if pos_score[i] > neg_score[i]: #FP 
                    FP.append([ ids[i], neg_score[i], pos_score[i]])
                else: # TN 
                    TN.append([ ids[i], neg_score[i], pos_score[i]])
            
        names = ['TP', 'FP', 'TN', 'FN'] 
        data_columns = ['Id', 'Neg Score', 'Pos Score']
        for i, data in enumerate([TP, FP, TN, FN]):
            data = sorted(data,key=lambda x: x[1])
            df = pd.DataFrame(data, columns=data_columns)
            df.to_csv(os.path.join(directory, clf + "_" + names[i] + ".csv"), index=False)

def run_selector_classifier(selector, all_prediction, all_prediction_scores, test_index, key_str, results, 
                            base_clfs, base_clf_names,
                            X_train_tranformed, y_train, X_test_transformed, y_test, 
                            num_classes, multiclass, results_params,
                            compare_classifiers, misclassified):
    

    # run classification
    if len(X_train_tranformed[0]) > 0:
        results, temp_pred, temp_pred_scores = run_classifiers(results, base_clfs, base_clf_names,
                                                    X_train_tranformed, y_train, X_test_transformed, y_test, 
                                                    num_classes, multiclass, results_params,
                                                    test_index, compare_classifiers, misclassified)
        # map prediction score over to all_prediction_scores
        if num_classes == 2 or multiclass == 'ovr':
            for j in range(len(test_index)):
                for name in base_clf_names:
                    if name != 'SVC':
                        if key_str is None:
                            all_prediction_scores[name][test_index[j]] = temp_pred_scores[name][j]
                            all_prediction[name][test_index[j]] = temp_pred[name][j]
                        else:
                            all_prediction_scores[key_str][name][test_index[j]] = temp_pred_scores[name][j]
                            all_prediction[key_str][name][test_index[j]] = temp_pred[name][j]
                            
    return results

def initial_all_prediction_scores(param_strs, base_clf_names, X):
    all_prediction_scores = dict()
    if param_strs is not None:
        for i in range(len(param_strs)): 
            prediction_scores = dict()
            for name in base_clf_names:
                if name != 'SVC':
                    prediction_scores[name] = [[0,0]] * len(X)
            all_prediction_scores[param_strs[i]] = prediction_scores
    else:
        for name in base_clf_names:
            if name != 'SVC':
                all_prediction_scores[name] = [[0,0]] * len(X)
    return all_prediction_scores

def init_misclassified_dict(base_clf_names):
    misclassified = dict()
    for name in base_clf_names:
        misclassified[name] = []
    return misclassified

def EmsembleFSClassifierPipeline(X, y, groups, 
                                 base_clfs, base_clf_names,
                                selector, combines, thresholds, cv_size, patient_ids, 
                                directory=None, filename=None, multiclass='ovr', 
                                compare_classifiers=None):
    # set number classes
    num_classes = np.max(y) + 1

    # make subfolder
    new_dir = make_dir_if_not_exist(directory, filename)

    # initial results dataframe
    results = pd.DataFrame(columns=["base clf", "combine", "threshold", "num features", "accuracy", "precision", "recall", "auc", "f1"])
    
    # prediction scores
    param_strs = None
    if combines is not None:
        param_strs = [combines[i] + str(thresholds[i]) for i in range(len(combines))]
    all_prediction_scores = initial_all_prediction_scores(param_strs, base_clf_names, X)
    all_predictions = initial_all_prediction_scores(param_strs, base_clf_names, X)
    
    results_params = dict()

    # dict of misclassified example indices for each classifier
    misclassified = dict()

    selection_folds = dict()
    for param in param_strs:
        misclassified[param] = init_misclassified_dict(base_clf_names)
        selection_folds[param] = []
    
    CVSplitter = StratifiedKFold(n_splits=cv_size, random_state=None, shuffle=True)
    
    
    for train_index, test_index in CVSplitter.split(X, groups): # split based on group_number
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        groups_train = groups[train_index]

        # run feature selection
        if combines is not None:
            selector.fit(X_train, y_train, combine=False, groups=groups_train)
        else:
            selector.fit(X_train, y_train)

        if combines is not None:
            for i in range(len(combines)):
                results_params['combine'] = combines[i]
                results_params['threshold'] = thresholds[i]
                key_str = combines[i] + str(thresholds[i])

                selector.combine_rankings(combines[i], thresholds[i])
                selection_folds[key_str].append(selector.selection_indices)
                X_train_tranformed = selector.transform(X_train)
                X_test_transformed = selector.transform(X_test)
                
                results = run_selector_classifier(selector, all_predictions, all_prediction_scores, test_index, key_str, 
                                                  results, base_clfs, base_clf_names,
                                                X_train_tranformed, y_train, X_test_transformed, y_test,
                                                num_classes, multiclass, results_params,
                                                compare_classifiers, misclassified[key_str])

        else:
            selection_folds.append(selector.selection_indices)
            X_train_tranformed = selector.transform(X_train)
            X_test_transformed = selector.transform(X_test)
            results = run_selector_classifier(selector, all_prediction_scores, test_index, None, 
                                              results, base_clfs, base_clf_names,
                                            X_train_tranformed, y_train, X_test_transformed, y_test, 
                                            num_classes, multiclass, results_params,
                                            compare_classifiers, misclassified)
    
    if combines is not None:
        for i in range(len(combines)):
            
            key_str = combines[i] + str(thresholds[i])
            prediction_scores = all_prediction_scores[key_str]
            if new_dir is not None:
                create_scores_csv(num_classes, y, prediction_scores, patient_ids, os.path.join(new_dir, key_str))
    else:
        if new_dir is not None:
            create_scores_csv(num_classes, y, all_prediction_scores, patient_ids, os.path.join(new_dir, filename))
   
    if compare_classifiers is not None and compare_classifiers=='mcnemar':
        for param in misclassified.keys():
            compare_all(misclassified[param], len(X), os.path.join(new_dir, param))
   
    if new_dir is not None:
        results.groupby(['base clf', 'combine', 'threshold']).mean().to_csv(os.path.join(new_dir, filename + '_mean' + ".csv"))
        results.to_csv(os.path.join(new_dir, filename + ".csv"))
    
    return results, all_predictions, all_prediction_scores, selection_folds

def SelectorThresholdClassifierPipeline(X, y, groups,
                                        cv_size, feature_selector, 
                                        base_clfs, base_clf_names, 
                                        directory, filename):
    new_dir = make_dir_if_not_exist(directory, filename)

    num_classes = np.max(y) + 1
    
    results = pd.DataFrame(columns=["base clf", "num features", "accuracy", "precision", "recall", "auc", "f1"])
    CVSplitter = StratifiedKFold(n_splits=cv_size, random_state=None, shuffle=True)
    
    shuffle_splitter = ShuffleSplit(n_splits=5, test_size=.2)
    
    for train_index, test_index in CVSplitter.split(X, groups):
        X_dev, X_test = X[train_index], X[test_index]
        y_dev, y_test = y[train_index], y[test_index]
        group_dev, group_test = groups[train_index], groups[test_index]
        
        for train_index, test_index in shuffle_splitter.split(X_dev, group_dev):
            X_train, X_holdout = X[train_index], X[test_index]
            y_train, y_holdout = y[train_index], y[test_index]

            # Feature selection 
            feature_selector.fit(X_train, y_train)
            X_train_tranformed = feature_selector.transform(X_train)
            X_holdout_transformed = feature_selector.transform(X_holdout)
            X_test_transformed = feature_selector.transform(X_test)
            
            # threshold classifier
            for i in range(len(base_clfs)):
                clf_temp = base_clfs[i]
                if hasattr(clf_temp, 'reset'): # emsembles
                    clf = clf_temp.reset()
                else:
                    clf = base.clone(clf_temp)
                
                threshold_clf = ThresholdClassifier(clf, multilabel=False)
                threshold_clf.fit(X_train_tranformed, y_train)
                threshold_clf.optimize_threshold(X_holdout_transformed, y_holdout)
                y_pred = threshold_clf.predict(X_test_transformed)
                results = add_row(results, y_test, y_pred, base_clf_names[i], len(X_train_tranformed[0]), num_classes, None)
                for class_i in range(len(threshold_clf.thresholds)):
                    results.loc[len(results)-1, 'threshold class ' + str(class_i)] = threshold_clf.thresholds[class_i]
    
    if new_dir is not None:
        results.groupby(['base clf']).mean().to_csv(os.path.join(new_dir, filename + "_mean" + ".csv"))
        results.to_csv(os.path.join(new_dir, filename + ".csv"))
    return results

def FilterSelectorClassifierPipeline(X, y, groups,
                                     base_clfs, base_clf_names,
                                     cv_size, feature_sizes, selector, patient_ids, 
                                     directory=None, filename=None, multiclass='ovr',
                                     compare_classifiers=None):
    num_classes = np.max(y) + 1

    new_dir = make_dir_if_not_exist(directory, filename)
            
    results = pd.DataFrame(columns=["base clf", "num features", "accuracy", "precision", "recall", "auc", "f1"])
    CVSplitter = StratifiedKFold(n_splits=cv_size, random_state=None, shuffle=True)
    
    # prediction scores
    param_strs = [str(f_size) for f_size in feature_sizes]
    all_prediction_scores = initial_all_prediction_scores(param_strs, base_clf_names, X)
    all_predictions = initial_all_prediction_scores(param_strs, base_clf_names, X)

    # dict of misclassified example indices for each classifier 
    misclassified = dict()
    selection_folds = dict()
    for param in param_strs:
        misclassified[param] = init_misclassified_dict(base_clf_names)
        selection_folds[param] = []

    for train_index, test_index in CVSplitter.split(X, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        selector.fit(X_train, y_train)
        for f_size in feature_sizes:
            key_str = str(f_size)
            X_train_tranformed = selector.transform(X_train, f_size)
            X_test_transformed = selector.transform(X_test, f_size)
            selection_folds[key_str].append(selector.selected_indices[:f_size])
            # run classification
            results = run_selector_classifier(selector, all_predictions, all_prediction_scores, test_index, key_str, results, 
                            base_clfs, base_clf_names,
                            X_train_tranformed, y_train, X_test_transformed, y_test, 
                            num_classes, multiclass, None,
                            compare_classifiers, misclassified[key_str])
        
    for f_size in feature_sizes:
        key_str = str(f_size)
        prediction_scores = all_prediction_scores[key_str]
        if new_dir is not None:
            create_scores_csv(num_classes, y, prediction_scores, patient_ids, os.path.join(new_dir, 'f_size_' + key_str))
    
    if compare_classifiers is not None and compare_classifiers=='mcnemar':
        for param in misclassified.keys():
            compare_all(misclassified[param], len(X), os.path.join(new_dir, 'f_size_' + param))

    if new_dir is not None:
        results.groupby(['base clf', 'num features']).mean().to_csv(os.path.join(new_dir, filename + "_mean" + ".csv"))
        results.to_csv(os.path.join(new_dir, filename + ".csv"))
    return results, all_predictions, all_prediction_scores, selection_folds

def RF_FS_CV(X, y, groups,
             n_estimators, num_features, patient_ids, 
             directory=None, filename=None):
    # make subfolder
    num_classes = np.max(y) + 1
    new_dir = make_dir_if_not_exist(directory, filename)
            
    results = pd.DataFrame(columns=["base clf", "num features", "accuracy", "precision", "recall", "auc", "f1"])
    CVSplitter = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
    
    prediction_scores = dict()
    prediction_scores['Random Forest'] = [[0,0]] * len(X)
    
    for train_index, test_index in CVSplitter.split(X, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        sel = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
        sel.fit(X_train, y_train)
        order = sel.feature_importances_.argsort().argsort()
        for f_size in num_features:
            selected_indices = []
            min_rank_select = len(order) - f_size
            for i in range(len(order)):
                if order[i] >= min_rank_select:
                    selected_indices.append(i)
            X_train_trans = X_train[:, selected_indices]

            clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
            clf.fit(X_train_trans, y_train)
            y_pred = clf.predict(X_test[:, selected_indices])
            #add_row(results, y_test, y_pred, base_clf, num_features, num_classes, prediction_proba):
            results = add_row(results, y_test, y_pred, 'RF', f_size, num_classes, prediction_scores['Random Forest'])
            
            y_scores = clf.predict_proba(X_test[:, selected_indices])
            for j in range(len(test_index)):
                prediction_scores['Random Forest'][test_index[j]] = y_scores[j]
    
    if new_dir is not None:
        create_scores_csv(num_classes, y, prediction_scores, patient_ids, os.path.join(new_dir, 'RF'))
        results.to_csv(os.path.join(new_dir, filename + ".csv"))
        results.groupby(['base clf', "num features"]).mean().to_csv(os.path.join(new_dir, filename + '_mean' + ".csv"))
    return results

def AGAFS(X, y, groups,
          aga, 
          base_clfs, base_clf_names, 
          cv_splits,
          patient_ids, directory=None, filename=None, 
          multiclass='ovr', compare_classifiers=None):
    
    num_classes = np.max(y) + 1
    new_dir = make_dir_if_not_exist(directory, filename)

    prediction_scores = initial_all_prediction_scores(None, base_clf_names, X)
    misclassified = init_misclassified_dict(base_clf_names)
            
    results = pd.DataFrame(columns=["base clf", "num features", "accuracy", "precision", "recall", "auc", "f1"])
    CVSplitter = StratifiedKFold(n_splits=cv_splits, random_state=None, shuffle=True)
    for train_index, test_index in CVSplitter.split(X, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        aga.fit(X_train, y_train)
        selection = aga.population[np.argmax(aga.fitnesses)]
        X_train_tranformed = X_train[:, selection]
        X_test_transformed = X_test[:, selection]
                        
        results, temp_pred_scores = run_classifiers(results, base_clfs, base_clf_names,
                                                    X_train_tranformed, y_train, X_test_transformed, y_test, 
                                                    num_classes, multiclass, None,
                                                    test_index, compare_classifiers, misclassified)
        for j in range(len(test_index)):
            for name in base_clf_names:
                if name in prediction_scores.keys():
                    prediction_scores[name][test_index[j]] = temp_pred_scores[name][j]
    
    if new_dir is not None:
        create_scores_csv(num_classes, y, prediction_scores, patient_ids, os.path.join(new_dir, filename))
        results.groupby(['base clf']).mean().to_csv(os.path.join(new_dir, filename + "_mean" + ".csv"))
        results.to_csv(os.path.join(new_dir, filename + ".csv"))
    return results

# Baseline classifier 
def CV_baseline(X, y, groups,
                cv_splits, directory=None, filename=None):
    all_results = []
    clfs = [LogisticRegression(penalty='l2'), 
            SVC(), 
            AdaBoostClassifier(n_estimators=1000),
            RandomForestClassifier(n_estimators=1000)]

    CVSplitter = StratifiedKFold(n_splits=cv_splits, random_state=None, shuffle=True)
    for train_index, test_index in CVSplitter.split(X, groups):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        baseline_clfs = BaselineClassifiers(clfs)
        baseline_clfs.fit(X_train, y_train)
        y_preds = baseline_clfs.predict(X_test)
        all_results.append(baseline_clfs.get_scores(y_test, y_preds))
    all_results = pd.concat(all_results, axis=0)
    if directory is not None:
        all_results.to_csv(directory + filename)
        all_results.groupby('classifer').mean().to_csv(os.path.join(directory, filename + ".csv"))
    return all_results

from statsmodels.stats.contingency_tables import mcnemar

def intersection_size(lst1, lst2): 
    count = 0
    for value in lst1:
        if value in lst2:
            count+=1
    return count 

def reject_null(misclf, total_examples, clf_name1, clf_name2):
    size = intersection_size(misclf[clf_name1], misclf[clf_name2])
    a = [[0,0], [0,0]]
    # misclassified by both
    a[0][0] = size
    # misclassified by A
    a[0][1] = len(misclf[clf_name1]) - size
    # misclassified by B
    a[1][0] = len(misclf[clf_name2]) - size
    # not misclassified by A or B
    a[1][1] = total_examples- a[0][0] - a[0][1] - a[1][0]
    result = mcnemar(a, exact=True)
    #print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    #alpha = 0.05
    print(clf_name1 + " v " + clf_name2 + " " + str(result.pvalue))
    return result.pvalue
    #if result.pvalue > alpha:
    #    return False
    #else:
    #    return True 
    
def compare_all(misclf, total_count, new_dir):
    clf_names = list(misclf.keys())
    sig_results = np.zeros([len(clf_names), len(clf_names)])
    for i, name in enumerate(clf_names):
        for j in range(i+1, len(clf_names)):
            result = reject_null(misclf, total_count, name, clf_names[j])
            sig_results[i][j] = result
    print(sig_results)
    # write results
    df = pd.DataFrame(sig_results, columns=clf_names)
    df['clf'] = clf_names
    if new_dir is not None:
        df.to_csv(os.path.join(new_dir, 'compare_clf.csv'), index=False)
    return sig_results