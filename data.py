import pandas as pd
import numpy as np
import sklearn as sk
import collections

def read_data(train_path,val_path,test_path, preprocess=True, save=False, series_length=7):
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
			test.to_csv(train_path.split('.csv')[0] + '_processed.', index=False)
	else:
		train = pd.read_csv(train_path)
		val = pd.read_csv(val_path)
		test = pd.read_csv(test_path)

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
		if col != 'score':
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


def get_series(df):
	pass