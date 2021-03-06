import pandas as pd
import numpy as np
import sklearn as sk
import collections
from imblearn.combine import SMOTEENN
import random

def read_data(train_path,val_path,test_path, preprocess=False, save=False, series_length=7):
	"""
	Read in train, val and test data sets. If preprocess == True then the
	preprocessing is performed from the raw data sets. If save == True
	saves the processed data sets for reuse.
	"""
	if preprocess:
		train = read_train_data(train_path, series_length)
		val = read_val_data(val_path,series_length)
		test = read_test_data(test_path,series_length)
		if save:
			train.to_csv(train_path.split('.csv')[0] + '_processed.csv', index=False)
			val.to_csv(val_path.split('.csv')[0] + '_processed.csv', index=False)
			test.to_csv(test_path.split('.csv')[0] + '_processed.csv', index=False)
	else:
		train = pd.read_csv(train_path)
		val = pd.read_csv(val_path)
		test = pd.read_csv(test_path)
	# return None, val, test
	return train, val, test

def read_train_data(path, series_length):
	"""
	Read in the training set and return a pre-processed version.
	"""
	train = pd.read_csv(path)
	train = preprocess(train, series_length)
	return train

def read_val_data(path, series_length):
	"""
	Read in the training set and return a pre-processed version.
	"""
	val = pd.read_csv(path)
	val = preprocess(val, series_length)
	return val

def read_test_data(path, series_length):
	"""
	Read in the training set and return a pre-processed version.
	"""
	test = pd.read_csv(path)
	test = preprocess(test, series_length)
	return test

def preprocess(df, series_length=7):
	"""
	Perform preprocessing steps on the given dataframe.
	"""
	df = df.drop(columns=['fips', 'date'])
	df = add_series_tag(df, series_length)	
	df = remove_irregular_series(df, series_length)
	df.score = round_scores(df)
	df = min_max_scale(df)
	return df

def min_max_scale(df):
	"""
	Perform columnwise min-max scaling.
	"""
	for col in df:
		if col != 'score' and col!='series':
			df[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())
	return df

def add_series_tag(df, series_length=7):
	"""
	Sequentially label series within dataframe.
	"""
	series = []
	count = 0
	for i, row in df.iterrows():
		if pd.isnull(row['score']):
			series.append(count)
		else:
			series.append(count)
			count+=1
	df.insert(0, 'series',series)
	return df


def remove_irregular_series(df,series_length=7):
	"""
	Labels are assigned for a week, so must remove series that
	are not the correct length.
	"""
	remove_vals = []
	series_counts = df['series'].value_counts()
	for i, val in series_counts.iteritems():
		if val != series_length:
			remove_vals.append(i)
	df = df[~df.series.isin(remove_vals)]
	return df

def round_scores(df):
	"""
	Some labels were interpolated by the data provider, so they must be rounded
	to the nearest integer to be used as a label.
	"""
	return df.score.round()

def balance_data(df, rand_state=None, series_length=7):
	"""
	Use dataset balancer to balance the classes found in the dataset.
	Classes with high frequency will be under-sampled and classes with
	low frequency will be over-sampled.
	"""
	# sampler = SMOTEENN(sampling_strategy='all', random_state=rand_state, n_jobs=12)
	labels = df.pop('score').dropna().astype(int)
	min_count = labels.value_counts().min()
	labels = labels.tolist()
	# feature_length = df.shape[1]-1
	features = get_series(df)
	
	zipped = zip(features, labels)
	separated_list = [[] for i in range(6)]
	for f, l in zipped:
		separated_list[l].append([f,l])

	balanced_train = []
	for sl in separated_list:
		temp = [x for x in random.sample(sl, min_count)]
		for t in temp:
			balanced_train.append(t)

	x, y = zip(*balanced_train)

	return x, y

def get_series(df):
	"""
	Split up the time series in the dataframe.
	"""
	features = []
	series_list = df.series.unique()
	df = df.set_index(['series'])
	for s in series_list:
		f = df.loc[s].values.tolist()
		features.append(f)

	return features

def flatten_series(x, series_length):
	flattened = []

	for s in x:
		print(s)
		flattened.append([i for v in s for i in v ])
		print(flattened[-1])

	return flattened


def unflatten_series(x, feature_length):
	"""
	Unflatten a series that was flattened for dataset balancing.
	"""
	unflattened = []
	num_series = len(x)/feature_length

	for i in range(num_series):
		unflattened.append(x[num_series*feature_length:num_series*feature_length+feature_length])

	print(len(unflattened), len(unflattened[0]))

	return unflattened

def calculate_class_metrics(cm, classes=[0,1,2,3,4,5]):
	"""
	Calculate metrics for each class in data set.
	"""
	acc, prec, recall,f1 = [], [],[],[]
	cm = np.array(cm)
	for c in classes:
		tp = cm[c][c]
		fp = 0
		for i, v in enumerate(cm[c]):
			if i != c:
				fp+= v
		tn = 0
		for i in range(cm.shape[0]):
			for j in range(cm.shape[1]):
				if i != c and j != c:
					tn += cm[i][j]
		fn = 0
		for i, v in enumerate(cm.T[c]):
			if i != c:
				fn += v

		acc.append(((tp+tn)/(tp+tn+fp+fn)) if (tp+tn+fp+fn) > 0 else 0)
		prec.append((tp/(tp+fp)) if (tp+fp) > 0 else 0)
		recall.append((tp/(tp+fn)) if (tp + fn) > 0 else 0 )
		f1.append((2*(prec[-1]*recall[-1])/(prec[-1]+recall[-1])) if (prec[-1]+recall[-1]) > 0 else 0)

	return acc, prec, recall, f1

def calculate_average_metrics(acc, prec,recall, f1):
	"""
	Calculate average metrics for model.
	"""
	avg_acc = np.mean(acc)
	avg_prec = np.mean(prec)
	avg_recall = np.mean(recall)
	avg_f1 = np.mean(f1)

	return avg_acc, avg_prec, avg_recall, avg_f1

