from ..combine.combine_subset import *
from sklearn import base 

class EmsembleFS():
    def __init__(self, selector, splitter, combine='union', threshold=3):
        self.selector = selector
        self.splitter = splitter
        self.combine = combine
        self.threshold = threshold
    
    def get_rankings(self, selector):
        ## RFECV
        if hasattr(selector, 'ranking_'):
            return selector.ranking_
        
        if hasattr(selector, 'score_'):
            return selector.score_
        
        ## SelectFromModel(estimator=estimator)
        if hasattr(selector, 'estimator_'):
            if hasattr(selector.estimator_, 'coef_'):
                return selector.estimator_.coef_[0]
        return []
    
    def get_selections(self, selector):
        if hasattr(selector, 'support_'):
            return selector.support_
        if hasattr(selector, 'get_support'):
            return selector.get_support()
        return []
    
    def fit(self, X, y):
        self.rankings = []
        self.selections = []
        # split into train holdout
        for train_index, test_index in self.splitter.split(X, y):
            print("running FS")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # run fs on each subset
            selector = base.clone(self.selector)
            selector.fit(X_train, y_train)
            self.rankings.append(self.get_rankings(selector))
            self.selections.append(self.get_selections(selector))
        self.selections = np.array(self.selections)
        self.combine_rankings()
        
    def combine_rankings(self):
        if self.combine == 'union':
            self.selection_indices, self.selection, self.remove = union(np.array(self.selections))
        elif self.combine == 'intersection':
            self.selection_indices, self.selection, self.remove = intersection(np.array(self.selections))
        elif self.combine == 'vote-threshold':
            self.selection_indices, self.selection, self.remove = threshold_union(np.array(self.selections), self.threshold)
        elif self.combine == 'min-rank':
            self.selection_indices, self.selection, self.remove = threshold_score(self.rankings, self.combine, self.threshold)
        elif self.combine == 'median-rank':
            self.selection_indices, self.selection, self.remove = threshold_score(self.rankings, self.combine, self.threshold)
        elif self.combine == 'mean-rank':
            self.selection_indices, self.selection, self.remove = threshold_score(self.rankings, self.combine, self.threshold)
        elif self.combine == 'gmean-rank':
            self.selection_indices, self.selection, self.remove = threshold_score(self.rankings, self.combine, self.threshold)

    def transform(self, X):
        return np.delete(X, self.remove, axis=1)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return np.delete(X, self.remove, axis=1)