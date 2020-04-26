from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score 
from sklearn import base
import numpy as np
 
class ThresholdClassifier():
    def __init__(self, clf, multilabel=False):
        self.clf = clf
        self.multilabel = multilabel
        
    def fit(self, X, y):
        self.clf = base.clone(self.clf)
        self.clf.fit(X, y)
    
    def transform_proba(self, scores):
        comp_scores = np.array([1.0 - score for score in scores])
        all_scores = np.vstack((comp_scores, scores)).T
        return all_scores
    
    def optimize_threshold(self, X, y):
        y_scores = None
        if hasattr(self.clf, "predict_proba"):
            y_scores = self.clf.predict_proba(X)
        else:
            y_scores = self.clf.decision_function(X)
            if type(y_scores[0]) is list or type(y_scores[0]) is np.ndarray:
                for column_i in range(len(y_scores[0])):
                    y_scores[:,column_i] = np.array(self.normalize(y_scores[:,column_i]))
            else:
                y_scores = np.array(self.normalize(y_scores))
            
            if not(type(y_scores[0]) is list or type(y_scores[0]) is np.ndarray):
                y_scores = self.transform_proba(y_scores)
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
        return 2*((precision*recall)/(precision+recall)) if precision+recall > 0.0 else 0
    
    def get_y_for_class(self, y, class_index):
        return np.array([1 if y[i] == class_index else 0 for i in range(len(y))])
        
    def get_thresholds_to_f_measure(self, y_actual, y_scores): 
        self.thresholds = []
        for class_index in range(len(y_scores[0])):
            y_score = y_scores[:,class_index]
            #print(class_index)
            #print(y_score)
            y_for_class = self.get_y_for_class(y_actual, class_index)
            print(y_for_class)
            fpr, tpr, thresholds = roc_curve(y_for_class, y_score, drop_intermediate=True)
            f1 = []
            f1_max = 0
            f1_index = 0
            for k, threshold in enumerate(thresholds[1: len(thresholds)-1]):
                y_predicted = self.adjust_prediction(y_score, threshold)
                cur_f1_score = self.cal_f1_score(y_for_class, y_predicted, class_index)
                f1.append(cur_f1_score)
                if (cur_f1_score > f1_max):
                    f1_max = cur_f1_score 
                    f1_index = k
            #print(f1)
            #print(thresholds)
            self.thresholds.append(thresholds[f1_index + 1]) ## NEEDED TO ADJUST INDEX!!!
        
    def normalize(self, y_scores):
        return [(y_scores[i] - y_scores.min()) / (y_scores.max() - y_scores.min()) for i in range(len(y_scores))]

    def predict(self, X):
        y_scores = None
        if hasattr(self.clf, "predict_proba"):
            y_scores = self.clf.predict_proba(X)
        else:
            y_scores = self.clf.decision_function(X)
            if type(y_scores[0]) is list or type(y_scores[0]) is np.ndarray:
                for column_i in range(len(y_scores[0])):
                    y_scores[:,column_i] = np.array(self.normalize(y_scores[:,column_i]))
            else:
                y_scores = np.array(self.normalize(y_scores))
            
            if not(type(y_scores[0]) is list or type(y_scores[0]) is np.ndarray):
                y_scores = self.transform_proba(y_scores)
        
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
    alpha = 0.05
    if result.pvalue > alpha:
        return False
    else:
        return True 
    
def compare_all(misclf, total_count, new_dir):
    print(misclf)
    print(total_count)
    print(new_dir)
    clf_names = list(misclf.keys())
    sig_results = np.empty([len(clf_names), len(clf_names)], dtype='str')
    for i, name in enumerate(clf_names):
        for j in range(i+1, len(clf_names)):
            result = reject_null(misclf, total_count, name, clf_names[j])
            sig_results[i][j] = str(result)
        # write results
    df = pd.DataFrame(sig_results, columns=clf_names)
    df['clf'] = clf_names
    if new_dir is not None:
        df.to_csv(os.path.join(new_dir, 'compare_clf.csv'), index=False)
    print(sig_results)
    return sig_results

def compare_all_mcnemar(misclf_folds, new_dir):
    clf_names = list(misclf_folds[0].keys())
    if 'total' in clf_names:
        clf_names.remove('total')   
    dfs = []
    for fold in misclf_folds.keys():
        sig_results = compare_fold(misclf_folds[fold], clf_names)
        # write results
        df = pd.DataFrame(sig_results, columns=clf_names)
        df['clf'] = clf_names
        if new_dir is not None:
            df.to_csv(os.path.join(new_dir, 'compare_clf_fold_' + str(fold) + '.csv'), index=False)
        dfs.append(df)
    return dfs