from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score 
from sklearn import base
import numpy as np
 
class ThresholdClassifier():
    def __init__(self, clf, multilabel=True):
        self.clf = clf
        self.multilabel = multilabel
        
    def fit(self, X, y):
        self.clf = base.clone(self.clf)
        self.clf.fit(X, y)
    
    def optimize_threshold(self, X, y):
        y_scores = None
        if hasattr(self.clf, "predict_proba"):
            y_scores = self.clf.predict_proba(X)
        else:
            y_scores = self.normalize(self.clf.decision_function(X))
        self.get_thresholds_to_f_measure(y, y_scores)
        
    def adjust_prediction(self, scores, threshold):
        return [0 if scores[i] < threshold else 1 for i in range(len(scores))]
    
    def cal_confusion(self, y, y_pred, class_value):
        tp, fp, fn, tn = 0.0, 0.0, 0.0, 0.0
        for i in range(len(y)):
            if y[i] == class_value:
                if y_pred[i] == 1:
                    tp += 1
                else: 
                    fn += 1
            else:
                if y_pred[i] == 1:
                    fp += 1
                else:
                    tn += 1
        return tp, fp, fn, tn
    
    def cal_f1_score(self, y, y_pred, class_value):
        tp, fp, fn, tn = self.cal_confusion(y, y_pred, class_value)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return 2*((precision*recall)/(precision+recall))
        
    def get_thresholds_to_f_measure(self, y_actual, y_scores): 
        self.thresholds = []
        for class_index in range(len(y_scores[0])):
            y_score = y_scores[:,class_index]
            fpr, tpr, thresholds = roc_curve(y_actual, y_score, drop_intermediate=True)
            f1 = []
            f1_max = 0
            f1_index = 0
            for k, threshold in enumerate(thresholds[1: len(thresholds)-1]):
                y_predicted = self.adjust_prediction(y_score, threshold)
                cur_f1_score = self.cal_f1_score(y_actual, y_predicted, class_index)
                f1.append(cur_f1_score)
                if (cur_f1_score > f1_max):
                    f1_max = cur_f1_score 
                    f1_index = k
            self.thresholds.append(thresholds[f1_index + 1]) ## NEEDED TO ADJUST INDEX!!!
        
    def normalize(self, y_scores):
        return [(y_scores[i] - y_scores.min()) / (y_scores.max() - y_scores.min()) for i in range(len(y_scores))]

    def predict(self, X):
        if hasattr(self.clf, "predict_proba"):
            y_scores = self.clf.predict_proba(X)
        else:
            y_scores = self.normalize(self.clf.decision_function(X))
        
        predictions = []
        if self.multilabel:
            for class_index in range(len(y_scores[0])):
                y_score = y_scores[:,class_index]
                y_predicted = self.adjust_prediction(y_score, self.thresholds[class_index])
                predictions.append(y_predicted)
            return np.transpose(np.array(predictions))
        else:
            pos_class_index = 1
            y_score = y_scores[:, pos_class_index] # 1 is pos class by default
            predictions = self.adjust_prediction(y_score, self.thresholds[pos_class_index])
            return np.array(predictions)
    
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
    
    # y_pred - 1 if predicted of class else 0
    # y - actual class prediction (can be multiclass)
    def get_scores(self, y, y_pred, class_index=None):
        if class_index is None:
            return {'accuracy' : accuracy_score(y, y_pred), 
                    'precision': precision_score(y, y_pred),  
                    'recall': recall_score(y, y_pred), 
                    'auc': roc_auc_score(y, y_pred), 
                    'f1' : f1_score(y, y_pred)}
        else:
            tp, fp, fn, tn = self.cal_confusion(y, y_pred, class_index)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            adjusted_y = [1 if y[i] == class_index else 0 for i in range(len(y))] 
            return {'accuracy' : (tp + tn) / (tp + fp + fn + tn), 
                    'precision': precision,  
                    'recall': recall, 
                    'auc': roc_auc_score(adjusted_y, y_pred), 
                    'f1' : 2*((precision*recall)/(precision+recall))}