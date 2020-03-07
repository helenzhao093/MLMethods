# Emsemble
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
selector = RFECV(estimator, step=1, cv=5)
splitter = ShuffleSplit(n_splits=5, test_size=.2)
emsembleFS = EmsembleFS(selector, splitter, combine='vote-threshold', threshold=4)
emsembleFS.fit(X, y)
emsembleFS.selection_indices

array([ 3,  6,  7,  9, 10, 12, 20, 21, 22, 23, 26, 27])
```

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
selector = RFECV(estimator, step=1, cv=5)
shuffle_splitter = ShuffleSplit(n_splits=5, test_size=.2)
emsembleFS = EmsembleFS(selector, shuffle_splitter, combine='vote-threshold', threshold=4)

clf = RandomForestClassifier(n=1000)
CVSplitter = StratifiedKFold(n_splits=5)
for train_index, test_index in splitter.split(X, y):
    emsemble = HomogeneousEmsemble(shuffle_splitter, emsembleFS, clf, combine='min')
    emsemble.fit(X_dev,y_dev)
    y_pred = emclf.predict(X_test)
    results = results.append(emclf.get_scores(y_test, emclf.predict(X_test)), ignore_index=True)
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
X_train, X_holdout, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)
clf = ThresholdClassifier(LogisticRegression(), multiclass=True)
clf.fit(X_train, y_train)
clf.optimize_threshold(X_holdout, y_holdout)
clf.predict(X_test)
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

### Cross Validation Example with Feature Selection
- For each fold, divide the training dataset into N subsets.
- Run feature selection and generate a threshold classifier on each subset. 
- Results in N classifiers for each fold. 

```python
results = pd.DataFrame(columns=["accuracy", "precision", "recall", "auc", "f1"]) 
shuffle_splitter = ShuffleSplit(n_splits=5, test_size=.2)
CVSplitter = StratifiedKFold(n_splits=5)
for train_index, test_index in CVSplitter.split(X, y):
    X_dev, X_test = X[train_index], X[test_index]
    y_dev, y_test = y[train_index], y[test_index]
    for train_index, test_index in shuffle_splitter.split(X_dev, y_dev):
        X_train, X_holdout = X[train_index], X[test_index]
        y_train, y_holdout = y[train_index], y[test_index]

        # Feature selection 
        feature_selector = EmsembleFS(SelectFromModel(estimator= LogisticRegression()), shuffle_splitter, combine='vote-threshold', threshold=3)
        feature_selector.fit(X_train, y_train)

        clf = ThresholdClassifier(LogisticRegression())
        clf.fit(feature_selector.transform(X_train), y_train)
        clf.optimize_threshold(feature_selector.transform(X_holdout), y_holdout)
        predictions = clf.predict(feature_selector.transform(X_test))
        results = results.append(clf.get_scores(y_test, predictions[:,1]), ignore_index=True)

```