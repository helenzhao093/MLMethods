from ..combine.combine_prediction import * 
import pandas as pd 
import copy 
from sklearn import base 
import numpy as np 
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import label_binarize


class Emsemble():
    def __init__(self, split_train=None, classifier=None, selector=None, combine='majority-vote'):
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

    def fit(self, X, y, clf=None, splitter=None, selector=None):
        self.num_classes = np.max(y) + 1
        selector = self.selector if self.selector is not None else selector
        if selector is not None:
            if type(selector) is list or type(selector) is np.ndarray:
                pass
            elif selector.__class__.__name__ == 'EmsembleFS':
                selector = copy.copy(selector) 
            elif selector.__class__.__name__ == 'mRMR': 
                pass
            else:
                selector = base.clone(selector)
        
        clf = self.classifier if self.classifier is not None else clf
        splitter = self.split_train if self.split_train is not None else splitter 
        
        if splitter == None:
            # run on the whole X, y
            self.run(X, y, clf, selector)
        else:
            # split train into random subsets
            for train_index, test_index in splitter.split(X, y):
                X_train = X[train_index]
                y_train = y[train_index]
                self.run(X_train, y_train, clf, selector)
                
    def run(self, X, y, clf, selector):
        if selector is not None:
            if type(selector) is list or type(selector) is np.ndarray:
                X = X[:, selector]
            else:
                selector.fit(X, y)
                X = selector.transform(X)
            self.selectors.append(selector)
        
        clf = base.clone(clf) 
        clf.fit(X, y)
        self.emsemble.append(clf)

    def transform(self, index, X):
        selector = self.selectors[index]
        if type(selector) is list or type(selector) is np.ndarray:
            return X[:, selector]
        return self.selectors[index].transform(X)
                
    def predict(self, X, combine='majority-vote'):
        if self.combine == 'majority-vote':
            predictions, probas = self.combine_labels(X)
            self.prediction_proba = probas
            return predictions
        else:
            predictions, probas = self.combine_predictions(X)
            self.prediction_proba = probas
            return predictions
    
    def combine_labels(self, X):
        predict_probas = []
        labels = []
        for clf_index in range(len(self.emsemble)):
            clf = self.emsemble[clf_index]
            X_temp = X
            if len(self.selectors) > 0:
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
            clf = self.emsemble[clf_index]
            X_temp = X
            if len(self.selectors) > 0:
                X_temp = self.transform(clf_index, X)
            
            prob = None
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