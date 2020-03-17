import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr, f_oneway

# Maximum Relevance and Minimum Redundancy
# https://arxiv.org/pdf/1908.05376.pdf
def MI_Matrix(X):
    scores = []
    for i in range(len(X[0])):
        feature_scores = []
        for j in range(i, len(X[0])):
            feature_scores.append(1/len(X[0]) * mutual_info_score(X[:,i], X[:,j]))
        scores.append(feature_scores)
    return scores

# mutual information difference
def MID(X, y):
    scores = MI_Matrix(X)
    mid = []
    for i in range(len(X[0])):
        current_score = mutual_info_score(X[:,i], y)
        for j in range(1, len(scores[i])):
            # print(i,j)
            current_score -= scores[i][j]
        l = 1
        for k in range(i-1, -1, -1):
            # print(k,l)
            current_score -= scores[k][l]
            l += 1
        mid.append(current_score)
    return np.array(mid)

# mutual information quotient
def MIQ(X, y):
    scores = MI_Matrix(X)
    mid = []
    for i in range(len(X[0])):
        mi = mutual_info_score(X[:,i], y)
        q = 0
        for j in range(1, len(scores[i])):
            # print(i,j)
            q += scores[i][j]
        l = 1
        for k in range(i-1, -1, -1):
            # print(k,l)
            q += scores[k][l]
            l += 1
        mid.append(mi/q)
    return np.array(mid)

def pearson_matrix(X):
    scores = []
    for i in range(len(X[0])):
        feature_scores = []
        for j in range(i, len(X[0])):
            feature_scores.append(1/len(X[0]) * np.corrcoef(X[:,i], X[:,j])[0, 1] ) 
        scores.append(feature_scores)
    return scores

# ftest for relevance, pearson for redundancy  
def FCD(X, y):
    scores = pearson_matrix(X)
    fcd = []
    for i in range(len(X[0])):
        f_test_score, p_value = f_oneway(X[:,i], y)
        for j in range(1, len(scores[i])):
            # print(i,j)
            f_test_score -= scores[i][j]
        l = 1
        for k in range(i-1, -1, -1):
            # print(k,l)
            f_test_score -= scores[k][l]
            l += 1
        fcd.append(f_test_score)
    return np.array(fcd)

def FCQ(X, y):
    scores = pearson_matrix(X)
    fcq = []
    for i in range(len(X[0])):
        f_test_score, p_value = f_oneway(X[:,i], y)
        temp = 0.0
        for j in range(1, len(scores[i])):
            # print(i,j)
            temp += scores[i][j]
        l = 1
        for k in range(i-1, -1, -1):
            # print(k,l)
            temp += scores[k][l]
            l += 1
        fcq.append(f_test_score/temp)
    return np.array(fcq)

def RFCQ(X, y):
    RF_clf = RandomForestClassifier(n_estimators=500)
    RF_clf.fit(X,y)
    scores = pearson_matrix(X)
    
    rfcq = []
    for i in range(len(X[0])):
        current_score = RF_clf.feature_importances_[i]
        temp = 0.0
        for j in range(1, len(scores[i])):
            # print(i,j)
            temp += scores[i][j]
        l = 1
        for k in range(i-1, -1, -1):
            # print(k,l)
            temp += scores[k][l]
            l += 1
        rfcq.append(current_score/temp)
    return np.array(rfcq)

class mRMR():
    def __init__(self, k=10, score_func='MID'):
        self.k = k
        self.score_func = score_func
        
    def fit(self, X, y):
        if self.score_func == 'MID':
            self.scores = MID(X, y)
        elif self.score_func == 'MIQ':
            self.scores = MIQ(X, y)
        elif self.score_func == 'FCD':
            self.scores = FCD(X, y)
        elif self.score_func == 'FCQ':
            self.scores = FCQ(X, y)
        elif self.score_func == 'RFCQ':
            self.scores = RFCQ(X, y)
    
    def transform(self, X, k):
        num_features = len(X[0])
        order = self.scores.argsort().argsort()
        self.selected_indices = []
        if k == 'all' or k >= num_features:
            self.selected_indices = np.array([i for i in range(num_features)])
        else:
            min_rank_select = num_features - k
            for i in range(len(order)):
                if order[i] >= min_rank_select:
                    self.selected_indices.append(i)
            self.selected_indices = np.array(self.selected_indices)
        return X[:, self.selected_indices]

        