import sklearn as sk
import numpy as np
import pyts

def get_random_forest(n_trees=100, criterion="gini", max_depth=None):
	rf = sk.ensemble.RandomForestClassifier(n_trees, criterion, max_depth)
	return rf

def get_time_series_forest(n_trees=100, criterion="gini", max_depth=None):
	tsf = pyts.classification.TimeSeriesForest(n_trees, criterion, max_depth)
	
