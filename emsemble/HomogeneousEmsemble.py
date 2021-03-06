from ..combine.combine_prediction import * 
import pandas as pd 
import copy 
from sklearn import base 
import numpy as np 
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize


class HomogeneousEmsemble():
    def __init__(self, split_train, classifier, selector=None, combine='median'):
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

    def reset(self):
        self.results = pd.DataFrame(columns=["accuracy", "precision", "recall", "roc", "f1"])
        self.emsemble = []
        self.selectors = []
        self.delete_indices = []
        return self
    
    def fit(self, X, y, splitter=None, selector=None):
        self.num_classes = np.max(y) + 1
        if splitter == None:
            splitter = self.split_train
        # split train into random subsets
        for train_index, test_index in splitter.split(X, y):
            # run feature selection algorithm on train
            X_train, X_holdout = X[train_index], X[test_index]
            y_train, y_holdout = y[train_index], y[test_index]
            if self.selector is not None:
                selector = copy.copy(self.selector) if self.selector.__class__.__name__ == 'EmsembleFS' else base.clone(self.selector)
                selector.fit(X_train, y_train)
                X_train = selector.transform(X_train)
                self.selectors.append(selector)
            elif selector is not None:
                selector = copy.copy(self.selector) if self.selector.__class__.__name__ == 'EmsembleFS' else base.clone(self.selector)
                selector.fit(X_train, y_train)
                X_train = selector.transform(X_train)
                self.selectors.append(selector)
            # run classifier on transformed data 
            clf = base.clone(self.classifier) 
            clf.fit(X_train, y_train)
            self.emsemble.append(clf)
        
    def transform(self, index, X):
        X_temp = X
        if self.selector.__class__.__name__ == 'EmsembleFS':
            X_temp = np.delete(X, self.selectors[index].remove, axis=1)
        else:
            X_temp = self.selectors[index].transform(X)
        return X_temp
                
    def predict(self, X):
        if self.combine == 'majority-vote':
            predictions, probas = self.combine_labels(X)
            self.prediction_proba = probas
            return predictions
        else:
            predictions, probas = self.combine_predictions(X)
            self.prediction_proba = probas
            return predictions

    def predict_proba(self, X):
        return self.prediction_proba
    
    def transform_proba(self, scores):
        comp_scores = np.array([1.0 - score for score in scores])
        all_scores = np.vstack((comp_scores, scores)).T
        return all_scores
    
    def combine_labels(self, X):
        labels = []
        predict_probas = []
        for clf_index in range(len(self.emsemble)):
            proba = None
            clf = self.emsemble[clf_index]
            X_temp = X
            if self.selector is not None:
                X_temp = self.transform(clf_index, X)
            labels.append(clf.predict(X_temp))
        
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_temp)
            elif hasattr(clf, "decision_function"):
                proba = clf.decision_function(X_temp)
                
                if type(proba[0]) is list or type(proba[0]) is np.ndarray:
                    for column_i in range(len(proba[0])):
                        proba[:,column_i] = np.array(self.normalize(proba[:,column_i]))
                else:
                    proba = np.array(self.normalize(proba))
            
            if not(type(proba[0]) is list or type(proba[0]) is np.ndarray):
                proba = self.transform_proba(proba)
            predict_probas.append(proba)   
        
        predictions = majority_vote(labels)
        _, predict_probas = median_rule(predict_probas)
        return predictions, predict_probas
            
    def combine_predictions(self, X):
        predict_probas = []
        for clf_index in range(len(self.emsemble)):
            proba = None
            clf = self.emsemble[clf_index]
            X_temp = X
            if self.selector is not None:
                X_temp = self.transform(clf_index, X)

            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_temp)
                
            elif hasattr(clf, "decision_function"):
                proba = clf.decision_function(X_temp)
                if type(proba[0]) is list or type(proba[0]) is np.ndarray:
                    for column_i in range(len(proba[0])):
                        proba[:,column_i] = self.normalize(proba[:,column_i])
                else:
                    proba = self.normalize(proba)
            
            if not(type(proba[0]) is list or type(proba[0]) is np.ndarray):
                proba = self.transform_proba(proba)
                
            predict_probas.append(proba)   
    
        if self.combine == 'median':
            return median_rule(predict_probas)
        elif self.combine == 'sum':
            return sum_rule(predict_probas)
        elif self.combine == 'max':
            return max_rule(predict_probas)
        elif self.combine == 'min':
            return min_rule(predict_probas)
        elif self.combine == 'product':
            return product_rule(predict_probas)
    
    def get_scores(self, y, y_pred):
        average='binary'
        if self.num_classes > 2:
            average='micro'
        print(self.num_classes)
        auc_score = None
        if average == 'binary':
            auc_score = roc_auc_score(y, y_pred) 
        else:
            auc_score = []
            classes = range(self.num_classes)
            y_bin = label_binarize(y, classes=classes)
            for i in range(self.num_classes):
                y_temp = y_bin[:,i]
                auc_score.append(roc_auc_score(y_temp, self.prediction_proba[:, i]))
            auc_score = str(auc_score)

        return {'accuracy' : accuracy_score(y, y_pred), 
                'precision': precision_score(y, y_pred, average=average),  
                'recall': recall_score(y, y_pred, average=average), 
                'auc': auc_score, 
                'f1' : f1_score(y, y_pred, average=average)} 
    
    def score(self, X, y):
        prediction = self.predict(X)
        return self.get_scores(y, prediction)