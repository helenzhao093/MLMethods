import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.177.9052&rep=rep1&type=pdf
class MSVMRFE():
    def __init__(self, step=0.1, cv=None, num_runs=5):
        self.step = step
        self.num_runs = num_runs
        if cv is None:
            self.cv = StratifiedKFold(n_splits=5, random_state=None, shuffle=True)
        else:
            self.cv = cv
    
    def fit(self, X, y):
        final_rank = np.array([0.0] * len(X[0]))
        num_features = len(X[0])
        min_features = int(0.1 * num_features)
        self.num_drop_features = self.step if self.step >= 1 else self.step * num_features
        indices = np.array([i for i in range(len(X[0]))])
        num_iter = 0
        while len(X[0]) > min_features:
            weights = []
            
            # train t linear svm on subsets of X
            for i in range(self.num_runs):
                for train_index, test_index in self.cv.split(X, y):
                    X_train, _ = X[train_index], X[test_index]
                    y_train, _ = y[train_index], y[test_index]
                    svm = LinearSVC()
                    svm.fit(X_train, y_train)               
                    norm = np.linalg.norm(svm.coef_[0])
                    weights.append([w/norm for w in svm.coef_[0]])
            # final weight = mean / stddev 
            final_weights = np.array([np.mean([weights[i][j] for i in range(len(weights))]) / np.std([weights[i][j] for i in range(len(weights))]) 
                             for j in range(len(weights[0]))])
            # drop features
            indices, X = self.reduce(X, final_weights, self.num_drop_features, indices, final_rank, num_iter)
            num_iter += 1
        
        for i in indices:
            final_rank[i] = num_iter
        self.max_rank = num_iter
        print(final_rank)
        self.ranking = np.array(final_rank)
    
    def reduce(self, X, weights, k, indices, final_rank, num_iter):
        num_features = len(X[0])
        order = weights.argsort().argsort()
        selected_indices = []
        min_rank_select = k
        new_indices = []
        for i in range(len(order)):
            if order[i] >= min_rank_select:
                selected_indices.append(i)
                new_indices.append(indices[i])
            else:
                final_rank[indices[i]] = num_iter
                #final_coef[indices[i]] = weights[i]
        return new_indices, X[:, selected_indices]
    
    def transform(self, X, size):
        min_rank = self.max_rank - size/self.num_drop_features
        selected_indices = []
        for i in range(len(self.ranking)):
            if self.ranking[i] > min_rank:
                selected_indices.append(i)
        return X[:, selected_indices]