# Feature Selection
Emsembles are meta-estimators that fit a number of classifiers on various subsets of the dataset. 

## Emsemble Feature Selection
The Emsemble Feature Selection class runs the provided feature selection method on N subset of the dataset 
and generates N feature selection subsets and/or rankings. 
The N selections/rankings are combined to obtain a final feature selection. 
The class provides various methods of combinating feature subsets.

### Combination of Output Methods
#### Subset combination options: 
- union : the union of the subsets of selected features 
- intersection : the intersection of the subsets of selected features 
- vote-threshold : the features selected at least x amount of times

#### Ranking combination options
The feature rankings are combined, the features are sorted, and a threshold is set to obtain a subset of the features. 
The options to assign rank to a feature are:
- mean-rank: mean of the N ranking 
- min-rank: the minimum of the N rankings
- median-rank: the median of the N rankings 
- gmean-rank: geometric mean of the N rankings

### Parameters
- selector : the base feature selection algorithm
- splitter : method for generating subsets of the dataset 
- combine : method for combining feature rankings; options are listed above 
- threshold: if combine is a ranking option then threshold is set to obtain number of features. 
if combine is vote-threshold then threshold is the number of times a feature has to be selected to be in the final set. 

### Example
```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import ShuffleSplit

estimator = LogisticRegression()
selector = RFECV(estimator, step=0.1, cv=5)
splitter = ShuffleSplit(n_splits=5, test_size=.2)
emsembleFS = EmsembleFS(selector, splitter, combine='vote-threshold', threshold=4)
emsembleFS.fit(X, y)
emsembleFS.selection_indices
"""
array([ 3,  6,  7,  9, 10, 12, 20, 21, 22, 23, 26, 27])
"""
```
## Maximum Relevance and Minimum Redundancy (mRMR) Feature Selection Methods
Selects features considering both the relevance for predicting the target variable and the redundancy within the selected features. (https://arxiv.org/pdf/1908.05376.pdf)
### mRMR Variants
- MID (mutual information difference) :  Mutual information between feature and target variable to calculate relevance and mutual information between each pair of features to calculate redundancy. Difference is used to balance the relevance and redundancy.
- MIQ (mutual information quotient) :  Quotient used to balance the two.
- FCD (F-test correlation difference) : F-statistic to score relevance and Pearson correlation to score redundancy. Difference used to balance the two.
- FCQ (F-test correlation quotient): Quotient used to balance the two.
- RFCQ (Random Forest correlation quotient) : Random forest importance score to score relevance. Pearson correlation to score redundancy. Quotient used to balance. 

### Parameters
- score_func : the mRMR variant to used
- k : the number of features to select

### Example
```
selector = mRMR(score_func='MIQ', k=10)
selector.fit(X, y)
X_transformed = selector.transform(X)
```

# Classification
## Homogeneous Emsemble
The Homogeneous Emsemble runs the same feature selection method and classification algorithm on N subset of the dataset to generate N classifiers.

### Combination of Predictions
The outputs of the N classifiers are combined to obtain the final class labels. The prediction labels can be combined by majority vote meaning the final prediction is the one predicted by the majority of classifiers. If the classification methods that provide probablity, the prediction scores can be combined. 

The options of combining prediction labels/scores are:
- majority-vote : predict x as the majority class predicted by classifiers  
- product : predict x as class c if the product of prediction scores of c is the max among the classes
- sum : predict x as class c if the sum of prediction scores of c is the max amongl the classes
- max : predict x as class c if the max of prediction scores of c is the max among the classes
- min : predict x as class c if the min of prediction scores of c is the max among the classes
- median : predict x as class c if the median of prediction scores of c is the max among the classes

### Cross Validation Example
```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, ShuffleSplit

results = pd.DataFrame(columns=["accuracy", "precision", "recall", "auc", "f1"])
estimator = LogisticRegression()
selector = RFECV(estimator, step=0.1, cv=5)
shuffle_splitter = ShuffleSplit(n_splits=5, test_size=.2)

clf = LogisticRegression(penalty='l2')
CVSplitter = StratifiedKFold(n_splits=5)
for train_index, test_index in splitter.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # run feature selection
    emsembleFS = EmsembleFS(selector, splitter, combine='intersection')
    emsembleFS.fit(X_train, y_train)
    X_train_tranformed = emsembleFS.transform(X_train)
    X_test_transformed = emsembleFS.transform(X_test)
    
    # create homogeneous emsemble
    emsemble = HomogeneousEmsemble(shuffle_splitter, clf, combine='min')
    emsemble.fit(X_train_tranformed, y_train)
    y_pred = emsemble.predict(X_test_transformed)
    results = results.append(emsemble.get_scores(y_test,y_pred), ignore_index=True)
```

# Threshold Classifier
Threshold classifier fits the training dataset with the underlying classifier and calculates the optimal prediction score for thresold each class label that yields the highest classification performance on the holdout dataset. An example is predicted as class c if the prediction score for class c is above the threshold for class c. An example can be labeled as multiple classes.

### Simple Multilabel Example
- Divided the 3 class dataset into train, holdout, and test sets. 
- Train the underlying classifier on the training set. 
- Compute the optimal prediction threshold for each class on the holdout set. 
- Predict the class labels of the test set. predict returns a 2D array (n_samples, n_classes) - each row represents the subset of class labels for an example. 

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_dev, X_train, y_dev, y_test = train_test_split(X, y, test_size=0.3)
X_train, X_holdout, y_train, y_holdout = train_test_split(X_dev, y_dev, test_size=0.2)
clf = ThresholdClassifier(LogisticRegression(), multilabel=True)
clf.fit(X_train, y_train)
clf.optimize_threshold(X_holdout, y_holdout)
predictions = clf.predict(X_test)
"""
array([[0, 0, 1],
       [0, 0, 1],
       [1, 1, 0],
       [1, 1, 1]])
"""

# get the performance for label 1
clf.get_scores(y_test, predictions[:,1], 1))
"""
{'accuracy': 0.956140350877193, 
 'precision': 0.9565217391304348, 
 'recall': 0.9705882352941176, 
 'auc': 0.952685421994885, 
 'f1': 0.9635036496350365}
"""
```