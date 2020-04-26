from scipy.stats.mstats import gmean
import math 
import numpy as np

def union(selections):
    return threshold_union(selections, 1)

def intersection(selections):
    return threshold_union(selections, len(selections))

def threshold_union(selections, threshold):
    selection = [] 
    selection_indices = []
    #remove = []
    for i in range(len(selections[0])):
        if selections[:,i].sum() >= threshold:
            selection.append(True)
            selection_indices.append(i)
        else:
            selection.append(False)
            #remove.append(i)
    return np.array(selection_indices), np.array(selection) #, np.array(remove)

def get_scores(rankings, column_index):
    return np.array([rankings[row_index][column_index] for row_index in range(len(rankings))])

def threshold_score(rankings, scoring, threshold):
    ranking = None
    if scoring == 'min-rank':
        ranking = min_rank(rankings)
    elif scoring == 'median-rank':
        ranking = median_rank(rankings)
    elif scoring == 'mean-rank':
        ranking = mean_rank(rankings)
    elif scoring == 'gmean-rank':
        ranking = gmean_rank(rankings)
    selection = [] 
    selection_indices = []
    #remove = []
    num_features = threshold if threshold > 1.0 else math.ceil(threshold * len(ranking))
    for i in range(len(ranking)):
        if ranking[i] < num_features:
            selection.append(True)
            selection_indices.append(i)
        else:
            selection.append(False)
            #remove.append(i)
    return np.array(selection_indices), np.array(selection) #, np.array(remove)

def min_rank(rankings):
    agg_scores = []
    for i in range(len(rankings[0])):
        agg_scores.append(min(get_scores(rankings, i)))
    return rank_scores(np.array(agg_scores))

def median_rank(rankings):
    agg_scores = []
    for i in range(len(rankings[0])):
        agg_scores.append(np.median(get_scores(rankings, i)))
    return rank_scores(np.array(agg_scores))
        
def mean_rank(rankings):
    agg_scores = []
    for i in range(len(rankings[0])):
        agg_scores.append(np.mean(get_scores(rankings, i)))
    return rank_scores(np.array(agg_scores))

def gmean_rank(rankings):
    agg_scores = []
    for i in range(len(rankings[0])):
        agg_scores.append(gmean(get_scores(rankings, i)))
    return rank_scores(np.array(agg_scores))
    
def rank_scores(scores):
    return scores.argsort().argsort()