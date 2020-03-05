from ..combine.combine_prediction import * 
import pandas as pd 
import copy 
from sklearn import base 
import numpy as np 
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

class HomogeneousEmsemble():
    def __init__(self, split_test, split_train, selector, classifier, combine='median'):
        self.split_test = split_test
        self.split_train = split_train
        self.selector = selector
        self.classifier = classifier
        self.combine = combine
        self.results = pd.DataFrame(columns=["accuracy", "precision", "recall", "roc", "f1"])
        self.emsemble = []
        self.selectors = []
        self.delete_indices = []
        
    def normalize(self, y_scores):
        return [(y_scores[i] - y_scores.min()) / (y_scores.max() - y_scores.min()) for i in range(len(y_scores))]
    
    def fit(self, X, y):
        # split train into random subsets
        for train_index, test_index in self.split_train.split(X, y):
            # run feature selection algorithm on train
            X_train, X_holdout = X[train_index], X[test_index]
            y_train, y_holdout = y[train_index], y[test_index]
            selector = copy.copy(self.selector) if self.selector.__class__.__name__ == 'EmsembleFS' else base.clone(self.selector)
            selector.fit(X_train, y_train)
            X_train = selector.transform(X_train)
            self.selectors.append(selector)

            # run classifier on transformed data 
            clf = base.clone(self.classifier)
            clf.fit(X_train, y_train)
            self.emsemble.append(clf)
        
    def predict(self, X):
        predict_probas = []
        num_clf = len(self.emsemble)
        for clf_index in range(num_clf):
            clf = self.emsemble[clf_index]
            X_temp = X
            if self.selector.__class__.__name__ == 'EmsembleFS':
                X_temp = np.delete(X, self.selectors[clf_index].remove, axis=1)
            else:
                X_temp = self.selectors[clf_index].transform(X)
            if hasattr(clf, "predict_proba"):
                predict_probas.append(clf.predict_proba(X_temp))
            elif hasattr(clf, "decision_function"):
                predict_probas.append(self.normalize(clf.decision_function(X_temp)))
        return self.combine_predictions(predict_probas)
        
            
    def combine_predictions(self, predictions):
        if self.combine == 'median':
            return median_rule(predictions)
        elif self.combine == 'sum':
            return sum_rule(predictions)
        elif self.combine == 'max':
            return max_rule(predictions)
        elif self.combine == 'min':
            return min_rule(predictions)
        elif self.combine == 'product':
            return product_rule(predictions)
    
    def get_scores(self, y, y_pred):
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        return {'accuracy' : accuracy, 
                'precision': precision,  
                'recall': recall, 
                'auc': roc_auc, 
                'f1' : f1}        