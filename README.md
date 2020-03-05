# Emsemble Feature Selection

Emsembles are meta-estimators that fit a number of classifiers on various subsets of the dataset. 
Emsemble Feature Selection class runs the same feature selection method on N subset of the dataset 
and generates N feature selection subsets and/or rankings. 
The N selections/rankings are combined to obtain a final feature selection. 
The class provides various methods of combinating feature subsets.

## Combination of Output Methods
### Subset combination options: 
- union : the union of the subsets of selected features 
- intersection : the intersection of the subsets of selected features 
- vote-threshold : the features selected at least x amount of times

### Ranking combination options
The feature rankings are combined, the features are sorted, and a threshold is set to obtain a subset of the features. 
The options to assign rank to a feature are:
- mean-rank: mean of the N ranking 
- min-rank: the minimum of the N rankings
- median-rank: the median of the N rankings 
- gmean-rank: geometric mean of the N rankings

## Parameters
- selector : the base feature selection algorithm
- splitter : method for generating subsets of the dataset 
- combine : method for combining feature rankings; options are listed above 
- threshold: if combine is a ranking option then threshold is set to obtain number of features. 
if combine is vote-threshold then threshold is the number of times a feature has to be selected to be in the final set. 

## Example
```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV

estimator = LogisticRegression()#RandomForestClassifier()
selector = RFECV(estimator, step=1, cv=5)
splitter = ShuffleSplit(n_splits=5, test_size=.2)
emsembleFS = EmsembleFS(selector, splitter, combine='vote-threshold', threshold=4)
emsembleFS.fit(X, y)
emsembleFS.selection_indices

array([ 3,  6,  7,  9, 10, 12, 20, 21, 22, 23, 26, 27])
```