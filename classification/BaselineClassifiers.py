from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score 
from sklearn.model_selection import StratifiedKFold
from sklearn import base 
import pandas as pd

class BaselineClassifiers():
    def __init__(self, clfs):
        self.init_clfs = clfs
        self.results = pd.DataFrame(columns=["classifier", "fold", "accuracy", "precision", "recall", "roc", "f1"])
        
    def fit(self, X, y):
        self.clfs = []
        for clf in self.init_clfs:
            clf = base.clone(clf)
            clf.fit(X, y)
            self.clfs.append(clf)

    def predict(self, X):
        predictions = []
        for clf in self.clfs:
            predictions.append(clf.predict(X))
        return predictions

    def get_scores(self, y, y_preds):
        results = pd.DataFrame(columns=["classifer", "accuracy", "precision", "recall", "roc", "f1"])
        for i, y_pred in enumerate(y_preds):
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            auc = roc_auc_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            row = [self.clfs[i].__class__.__name__, accuracy, precision, recall, auc, f1]
            results.loc[len(results)] = row
        return results
    
    def score(self, X, y):
        predictions = self.predict(X)
        return self.get_scores(y, predictions)