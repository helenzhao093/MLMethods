import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import train_test_split

# https://arxiv.org/pdf/1711.08421.pdf
class Relief():
    def __init__(self, train_size=0.8):
        self.train_size = train_size
    
    def fit(self, X, y):
        weights = np.array([0.0] * len(X[0]))
        max_feature_values = np.array([max([X[i][j] for i in range(len(X))]) for j in range(len(X[0]))])
        min_feature_values = np.array([min([X[i][j] for i in range(len(X))]) for j in range(len(X[0]))])
        indices = np.arange(len(X))
        _, _, _, _, indices, _= train_test_split(X, y, indices, train_size=self.train_size)
        for ex_i in indices:
            min_hit_index, min_miss_index = self.nearest(ex_i, X, y)
            
            # update weights
            for feature_i in range(len(X[0])):
                weights[feature_i] -= (X[ex_i][feature_i] - X[min_hit_index][feature_i])/(max_feature_values[feature_i] - min_feature_values[feature_i])/len(indices) 
                weights[feature_i] += (X[ex_i][feature_i] - X[min_miss_index][feature_i])/(max_feature_values[feature_i] - min_feature_values[feature_i])/len(indices)
        self.weights = weights
        
    def nearest(self, i, X, y):
        x = X[i]
        y_actual = y[i]
        min_hit = -1
        min_hit_index = -1
        min_miss = -1
        min_miss_index = -1
        for j in range(len(X)):
            if j != i and y[j] == y_actual:
                cur_dist = distance.cityblock(X[j], x)
                if min_hit < 0: 
                    min_hit = cur_dist
                    min_hit_index = j
                elif cur_dist < min_hit:
                    min_hit = cur_dist
                    min_hit_index = j
            if j != i and y[j] != y_actual:
                cur_dist = distance.cityblock(X[j], x)
                if min_miss < 0: 
                    min_miss = cur_dist
                    min_miss_index = j
                elif cur_dist < min_hit:
                    min_miss = cur_dist
                    min_miss_index = j
        return min_hit_index, min_miss_index
    
    def transform(self, X, k):
        num_features = len(X[0])
        # lower the range, the lower the weight 
        order = self.weights.argsort().argsort()
        self.selected_indices = []
        max_weight_rank = k
        for i in range(len(order)):
            if order[i] < max_weight_rank:
                self.selected_indices.append(i)
        self.selected_indices = np.array(self.selected_indices)
        return X[:, self.selected_indices]