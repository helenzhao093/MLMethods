import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr, f_oneway
from scipy.stats import entropy
from math import log, e
import random
import time
import heapq
import math

# Maximum Relevance and Minimum Redundancy
# https://arxiv.org/pdf/1908.05376.pdf
def MI_Matrix(X):
    #start_time = time.time()
    num_features = len(X[0])
    scores = []
    for i in range(num_features):
        feature_scores = []
        for j in range(i, num_features):
            feature_scores.append(1/num_features * mutual_info_score(X[:,i], X[:,j]))
        scores.append(feature_scores)
    #end_time = time.time()
    #print("Run time = {}".format(end_time - start_time))
    return scores

# mutual information difference
def MID(X, y, k):
    num_features = len(X[0])
    start_feature_index = random.randint(0, num_features-1)
    selected_indices = set()
    selected_indices.add(start_feature_index)
    mi_scores = [mutual_info_score(X[:,i], y) for i in range(num_features)]
    
    mi_score_matrix = np.zeros(( num_features , num_features))
    for _ in range(k-1):
        start_time = time.time()
        temp_scores = []
        for i in range(num_features):
            if i in selected_indices:
                temp_scores.append(-float('inf'))
            else:
                score = mi_scores[i]
                diff = 0
                for j in selected_indices:
                    if j > i:
                        if mi_score_matrix[i][j] == 0:
                            mi_score_matrix[i][j] = np.corrcoef(X[:,i], X[:,j])[0, 1]
                        diff += mi_score_matrix[i][j]
                    else:
                        if mi_score_matrix[j][i] == 0:
                            mi_score_matrix[j][i] = np.corrcoef(X[:,i], X[:,j])[0, 1]
                        diff += mi_score_matrix[j][i]
                temp_scores.append(score - diff/len(selected_indices))
        selected_indices.add(np.argmax(np.array(temp_scores)))
        end_time = time.time()
        #print("Run time = {}".format(end_time - start_time))
    return selected_indices

# mutual information quotient
def MIQ(X, y, k):
    start_time = time.time()
    num_features = len(X[0])
    
    start_feature_index = random.randint(0, num_features-1)
    selected_indices = set()
    selected_indices.add(start_feature_index)
    mi_scores = [mutual_info_score(X[:,i], y) for i in range(num_features)]
    
    mi_score_matrix = np.zeros(( num_features , num_features))
    for _ in range(k-1):
        start_time = time.time()
        #print ('selecting')
        temp_scores = []
        for i in range(num_features):
            if i in selected_indices:
                temp_scores.append(-float('inf'))
            else:
                mi_score = mi_scores[i]
                q = 0
                for j in selected_indices:
                    if j > i:
                        if mi_score_matrix[i][j] == 0:
                            mi_score_matrix[i][j] = np.corrcoef(X[:,i], X[:,j])[0, 1]
                        q += mi_score_matrix[i][j]
                    else:
                        if mi_score_matrix[j][i] == 0:
                            mi_score_matrix[j][i] = np.corrcoef(X[:,i], X[:,j])[0, 1]
                        q += mi_score_matrix[j][i]
                temp_scores.append(mi_score/(q/len(selected_indices)))
        selected_indices.add(np.argmax(np.array(temp_scores)))
        end_time = time.time()
        #print("Run time = {}".format(end_time - start_time))
    return selected_indices

def pearson_matrix(X):
    start_time = time.time()
    scores = []
    for i in range(len(X[0])):
        feature_scores = []
        for j in range(i, len(X[0])):
            feature_scores.append(1/len(X[0]) * np.corrcoef(X[:,i], X[:,j])[0, 1] ) 
        scores.append(feature_scores)
    end_time = time.time()
    # print("Run time = {}".format(end_time - start_time))
    return scores

# ftest for relevance, pearson for redundancy  
def FCD(X, y, k):
    num_features = len(X[0])
    f_test_scores = [f_oneway(X[:,i], y)[0] for i in range(num_features)]
    
    start_feature_index = random.randint(0, num_features-1)
    selected_indices = set()
    selected_indices.add(start_feature_index)
    
    pearson_score_matrix = np.zeros(( num_features , num_features))
    for _ in range(k-1):
        start_time = time.time()
        temp_scores = []
        for i in range(num_features):
            if i in selected_indices:
                temp_scores.append(-float('inf'))
            else:
                f_test_score = f_test_scores[i]
                diff = 0
                for j in selected_indices:
                    # pearson score
                    if j > i:
                        if pearson_score_matrix[i][j] == 0:
                            pearson_score_matrix[i][j] = np.corrcoef(X[:,i], X[:,j])[0, 1]
                        diff += pearson_score_matrix[i][j]
                    else:
                        if pearson_score_matrix[j][i] == 0:
                            pearson_score_matrix[j][i] = np.corrcoef(X[:,i], X[:,j])[0, 1]
                        diff += pearson_score_matrix[j][i]
                    #diff += np.corrcoef(X[:,i], X[:,j])[0, 1]
                temp_scores.append(f_test_score - diff/len(selected_indices))
        end_time = time.time()
        #print("Run time = {}".format(end_time - start_time))
        selected_indices.add(np.argmax(np.array(temp_scores)))
    return selected_indices

def FCQ(X, y, k):
    num_features = len(X[0])
    f_test_scores = [f_oneway(X[:,i], y)[0] for i in range(num_features)]
    
    start_feature_index = random.randint(0, num_features-1)
    selected_indices = set()
    selected_indices.add(start_feature_index)
    
    pearson_score_matrix = np.zeros(( num_features , num_features))
    
    for _ in range(k-1):
        start_time = time.time()
        temp_scores = []
        for i in range(num_features):
            if i in selected_indices:
                temp_scores.append(-float('inf'))
            else:
                f_test_score = f_test_scores[i]
                q = 0
                for j in selected_indices:
                    # pearson score 
                    if j > i:
                        if pearson_score_matrix[i][j] == 0:
                            pearson_score_matrix[i][j] = np.corrcoef(X[:,i], X[:,j])[0, 1]
                        q += pearson_score_matrix[i][j]
                    else:
                        if pearson_score_matrix[j][i] == 0:
                            pearson_score_matrix[j][i] = np.corrcoef(X[:,i], X[:,j])[0, 1]
                        q += pearson_score_matrix[j][i]
                temp_scores.append(f_test_score/(q/len(selected_indices)))
        end_time = time.time()
        #("Run time = {}".format(end_time - start_time))
        selected_indices.add(np.argmax(np.array(temp_scores)))
    return selected_indices

def RFCQ(X, y, k):
    num_features = len(X[0])
    RF_clf = RandomForestClassifier(n_estimators=500)
    RF_clf.fit(X,y)
    
    start_feature_index = random.randint(0, num_features-1)
    selected_indices = set()
    selected_indices.add(start_feature_index)
    
    pearson_score_matrix = np.zeros(( num_features , num_features))
    
    for _ in range(k-1):
        start_time = time.time()
        temp_scores = []
        for i in range(num_features):
            if i in selected_indices:
                temp_scores.append(-float('inf'))
            else:
                rf_score = RF_clf.feature_importances_[i]
                q = 0
                for j in selected_indices:
                    # pearson score 
                    if j > i:
                        if pearson_score_matrix[i][j] == 0:
                            pearson_score_matrix[i][j] = np.corrcoef(X[:,i], X[:,j])[0, 1]
                        q += pearson_score_matrix[i][j]
                    else:
                        if pearson_score_matrix[j][i] == 0:
                            pearson_score_matrix[j][i] = np.corrcoef(X[:,i], X[:,j])[0, 1]
                        q += pearson_score_matrix[j][i]
                temp_scores.append(rf_score/(q/len(selected_indices)))
        end_time = time.time()
        #print("Run time = {}".format(end_time - start_time))
        selected_indices.add(np.argmax(np.array(temp_scores)))
    return selected_indices

def SU(x_i, x_j):
    return 2 * (mutual_info_score(x_i, x_j) / (entropy2(x_i) + entropy2(x_j)))

def entropy1(labels, base=None):
    value,counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

def entropy2(labels, base=None):
    """ Computes entropy of label distribution. """
    n_labels = len(labels)
    if n_labels <= 1:
        return 0
    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)
    return ent

def CFS(X, y, k):
    num_features = len(X[0])
    # rcf
    merit_feature_class = [SU(X[:,i], y) for i in range(num_features)]
    # rff 
    rff = []
    for i in range(num_features):
        scores = []
        for _ in range(0, i+1):
            scores.append(0)
        for j in range(i+1, num_features):
            scores.append(SU(X[:,i], X[:,j]))
        rff.append(scores)
        
    start_index = np.argmax(np.array(merit_feature_class))
    selected_indices = set()
    selected_indices.add(start_index)
    
    sum_cf = merit_feature_class[start_index]
    sum_ff = 0
    overall_merit = 0
    num_ff = 0
    for _ in range(k-1):
        start_time = time.time()
        current_merits = []
        num_ff += len(selected_indices)
        for i in range(num_features):
            if i not in selected_indices:
                sum_cf += merit_feature_class[i]
                added_sum_ff = 0
                for j in selected_indices:
                    if j > i:
                        added_sum_ff += rff[i][j]
                    else:
                        added_sum_ff += rff[j][i]
                sum_ff += added_sum_ff
                merit = sum_cf / math.sqrt(len(selected_indices)+1 + (( sum_ff/num_ff) * len(selected_indices) * len(selected_indices)+1))
                current_merits.append(merit)
                sum_cf -= merit_feature_class[i]
                sum_ff -= added_sum_ff
            else:
                current_merits.append(-float('inf'))
        #print(current_merits)
        selected_indices.add(np.argmax(np.array(current_merits)))
        end_time = time.time()
        #print("Run time = {}".format(end_time - start_time))
    return selected_indices

def num_ff_interactions(k):
    count = 0
    for i in range(k):
        count += i
    return count 

def CFS_heuristic_search(X, y, k):
    num_features = len(X[0])
    # rcf
    cf_scores = [SU(X[:,i], y) for i in range(num_features)]
    # rff 
    ff_scores = []
    for i in range(num_features):
        scores = []
        for _ in range(0, i+1):
            scores.append(0)
        for j in range(i+1, num_features):
            scores.append(SU(X[:,i], X[:,j]))
        ff_scores.append(scores)
    
    start_index = random.randint(0, len(cf_scores)-1)
    selected_indices = set()
    selected_indices.add(start_index)
    h = [] 
    # total score cf score, ff score, 
    heapq.heappush(h, (-cf_scores[start_index], (cf_scores[start_index], 0, selected_indices)))
    while len(selected_indices) < k:
        #start_time = time.time()
        merit, info = heapq.heappop(h)
        #print(info)
        cf, ff, cur_indices = info[0], info[1], info[2]
        for i in range(num_features):
            if i not in cur_indices: # not already selected
                cf += cf_scores[i]
                added_ff = 0
                for j in cur_indices:
                    if j > i:
                        added_ff += ff_scores[i][j]
                    else:
                        added_ff += ff_scores[j][i]
                ff += added_ff
                num_ff = num_ff_interactions(len(cur_indices)+1)
                merit = cf/math.sqrt(len(cur_indices)+1 + (( cf/num_ff) * len(cur_indices) * len(cur_indices)+1))
                temp = cur_indices.copy()
                temp.add(i)
                heapq.heappush(h, (-merit, (cf, ff, temp)))
                cf -= cf_scores[i]
                ff -= added_ff
        #end_time = time.time()
        #print("Run time = {}".format(end_time - start_time))
        selected_indices = cur_indices 
    return selected_indices


class mRMR():
    def __init__(self, k=10, score_func='MID'):
        self.k = k
        self.score_func = score_func
        
    def fit(self, X, y, k=None):
        if k is None:
            k = self.k 
        # select random feature to start search
        if self.score_func == 'MID':
            self.selected_indices = MID(X, y, k)
        elif self.score_func == 'MIQ':
            self.selected_indices = MIQ(X, y, k)
        elif self.score_func == 'FCD':
            self.selected_indices = FCD(X, y, k)
        elif self.score_func == 'FCQ':
            self.selected_indices = FCQ(X, y, k)
        elif self.score_func == 'RFCQ':
            self.selected_indices = RFCQ(X, y, k)
        elif self.score_func == 'CFS':
            self.selected_indices = CFS(X, y, k)
        elif self.score_func == 'CFS Heuristic':
            self.selected_indices = CFS_heuristic_search(X, y, k)
    
    def transform(self, X):
        return X[:, list(self.selected_indices)]

        