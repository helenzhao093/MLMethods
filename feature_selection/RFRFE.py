import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score 

class RFRFE():
    def __init__(self, n_estimators=1000, step=0.1, scoring="f1"):
        self.n_estimators = n_estimators
        self.step = step
        self.scoring = scoring
        self.results = pd.DataFrame(columns=["num_features", "oob", "precision", "recall", "auc", "f1"])
        self.selectors = []
        self.selections = []
        
    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        min_features = self.step * len(X_train[0])
        num_features = len(X_train[0])
        self.selections.append(np.array([i for i in range(len(X_train[0]))]))
        while len(X_train[0]) > min_features:
            rf = RandomForestClassifier(oob_score=True, n_estimators=self.n_estimators)
            sel = SelectFromModel(rf, max_features=num_features, threshold=-np.inf)
            sel.fit(X_train, y_train)
            
            indices = sel.transform([[i for i in range(len(X_train[0]))]])[0]
            self.selections.append(indices)
            
            X_train = sel.transform(X_train)
            self.score_rf(sel, X_test, y_test)
            X_test = sel.transform(X_test)
            num_features = int((1.0 - self.step) * num_features)
            self.selectors.append(sel)
        max_index = self.get_max(self.scoring)
        self.selected_indices = self.selections[max_index]


    def score_rf(self, sel, X, y):
        y_pred = sel.estimator_.predict(X)
        f1 = f1_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        auc = roc_auc_score(y, y_pred)
        row = [len(X[0]), sel.estimator_.oob_score_, precision, recall, auc, f1]
        self.results.loc[len(self.results)] = row
        
    def transform(self, X):
        return X[:, self.selected_indices]
        
    def get_max(self, scoring):
        return max(self.results.loc[self.results[scoring] == self.results.max()[scoring]].index)