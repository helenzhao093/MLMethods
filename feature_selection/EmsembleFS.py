from ..combine.combine_subset import *
from sklearn import base 

class EmsembleFS():
    def __init__(self, selector=None, splitter=None, combine='union', threshold=3):
        self.selector = selector
        self.splitter = splitter
        self.combine = combine
        self.threshold = threshold
        self.rankings = []
        self.selections = []
    
    def get_rankings(self, selector):
        ## RFECV
        if hasattr(selector, 'ranking_'):
            return selector.ranking_
        
        if hasattr(selector, 'score_'):
            return selector.score_
        
        if hasattr(selector, 'feature_importances_'):
            return selector.feature_importances_
        
        ## SelectFromModel(estimator=estimator)
        if hasattr(selector, 'estimator_'):
            if hasattr(selector.estimator_, 'coef_'):
                return selector.estimator_.coef_[0]
        return None
    
    def get_selections(self, selector):
        if hasattr(selector, 'support_'):
            return selector.support_
        if hasattr(selector, 'get_support'):
            return selector.get_support()
        return None
    
    def add_selector(self, X, y, selector, splitter):
        for train_index, test_index in splitter.split(X, y):
            print("running FS")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            selector = base.clone(selector)
            selector.fit(X_train, y_train)
            self.rankings.append(self.normalize(self.get_rankings(selector)))
            self.selections.append(self.get_selections(selector))
            
    def add_filter_selector(self, X, y, selector, splitter):
        for train_index, test_index in splitter.split(X, y):
            print("running FS")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            selector.fit(X_train, y_train)
            if self.get_rankings(selector) is not None:
                self.rankings.append(self.normalize(self.get_rankings(selector)))
            if self.get_selections(selector) is not None:
                self.selections.append(self.get_selections(selector))
            
    def fit(self, X, y, combine=True, groups=None):
        self.rankings = []
        self.selections = []
        # split into train holdout
        groups_split = y if groups is None else groups
        for train_index, test_index in self.splitter.split(X, groups_split):
            print("running FS")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # run fs on each subset
            selector = base.clone(self.selector)
            selector.fit(X_train, y_train)
            self.rankings.append(self.get_rankings(selector))
            self.selections.append(self.get_selections(selector))
        self.selections = np.array(self.selections)
        if combine == True:
            self.combine_rankings(self.combine, self.threshold)
        
    def normalize(self, y_scores):
        return np.array([(y_scores[i] - y_scores.min()) / (y_scores.max() - y_scores.min()) for i in range(len(y_scores))])
        
    def combine_rankings(self, combine, threshold):
        if combine == 'union':
            self.selection_indices, self.selection = union(np.array(self.selections))
        elif combine == 'intersection':
            self.selection_indices, self.selection = intersection(np.array(self.selections))
        elif combine == 'vote-threshold':
            self.selection_indices, self.selection = threshold_union(np.array(self.selections), threshold)
        elif combine == 'min-rank':
            self.selection_indices, self.selection = threshold_score(self.rankings, combine, threshold)
        elif combine == 'median-rank':
            self.selection_indices, self.selection  = threshold_score(self.rankings, combine, threshold)
        elif combine == 'mean-rank':
            self.selection_indices, self.selection  = threshold_score(self.rankings, combine, threshold)
        elif combine == 'gmean-rank':
            self.selection_indices, self.selection = threshold_score(self.rankings, combine, threshold)

    def transform(self, X):
        return X[:, self.selection_indices] #np.delete(X, self.remove, axis=1)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return X[:, self.selection_indices] #np.delete(X, self.remove, axis=1)