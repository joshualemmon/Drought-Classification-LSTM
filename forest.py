from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from pyts.classification import TimeSeriesForest
import pandas as pd
import data
import matplotlib.pyplot as plt
from sktime.classification.compose import TimeSeriesForestClassifier

def init_random_forest(n_trees=100, criterion="gini", max_depth=None):
	rf = RandomForestClassifier(n_trees, criterion, max_depth)
	return rf

def init_time_series_forest(n_trees=100, criterion="gini", max_depth=None, series_length=7):
	tsf = RandomForestClassifier(n_estimators=n_trees, criterion=criterion, max_depth=max_depth)
	return tsf

def train_random_forest(rf, train, val, balanced=False):
	"""
	Train random forest model 
	"""
	if balanced:
		temp = []
		labels = []
		for i, s in enumerate(train[0]):
			temp.append(s[-1])
			labels.append(train[1][i])
		train = temp
	else:
		train = train[~train['score'].isnull()]
		labels = train.pop('score')
		train = train.drop(columns=['series'])
	rf.fit(train, labels)
	print('Random Forest Trained')

	val_acc, val_prec, val_recall, val_f1 = test_random_forest(rf, val, True, balanced)
	print(f"Random Forest Trained\nVal Acc. {val_acc}\nVal Prec. {val_prec}\nVal Recall {val_recall}\nVal F1 {val_f1}")
	return rf

def train_time_series_forest(tsf, train, val, balanced=False):
	if balanced:
		labels = train[1]
		train = compute_window_vals(train[0])
	else:
		labels = train.pop('score').dropna().astype(int)
		train = compute_window_vals(data.get_series(train))
	tsf.fit(train, labels)
	print('Time Series Forest Trained')
	val_acc, val_prec, val_recall, val_f1 = test_time_series_forest(tsf, val.copy(True), True, balanced)
	print(f"Time Series Forest Trained\nVal Acc. {val_acc}\nVal Prec. {val_prec}\nVal Recall {val_recall}\nVal F1 {val_f1}")
	return tsf

def test_random_forest(rf, test, val=False, balanced=False):
	"""
	Test the trained random forest model and return calculated metrics.
	"""
	test = test[~test['score'].isnull()]
	labels = test.pop('score').astype(int)
	test = test.drop(columns=['series'])
	acc = 0
	preds = []
	for i, row in test.iterrows():
		pred = rf.predict([row])
		preds.append(int(pred))
	fig = plt.figure()
	cm = confusion_matrix(labels, preds, labels=[0,1,2,3,4,5])
	disp = ConfusionMatrixDisplay(cm, display_labels=['None', 'D0', 'D1', 'D2', 'D3', 'D4'])
	disp.plot()
	name = 'images/random_forest_cm'
	if val:
		name += '_val'
	else:
		name += '_test'
	if balanced:
		name+='_bal'
	plt.savefig(name+'.png', bbox_inches='tight')
	acc, prec, recall, f1 = data.calculate_class_metrics(cm)


	return acc, prec, recall, f1

def test_time_series_forest(tsf, test, val=False, balanced=False):
	labels = test.pop('score').dropna().tolist()
	test_series = compute_window_vals(data.get_series(test))
	preds = tsf.predict(test_series)
	cm = confusion_matrix(labels, preds, labels=[0,1,2,3,4,5])
	fig = plt.figure()
	disp = ConfusionMatrixDisplay(cm, display_labels=['None', 'D0', 'D1', 'D2', 'D3', 'D4'])
	disp.plot()
	name = 'images/time_series_forest_cm'
	if val:
		name += '_val'
	else:
		name += '_test'
	if balanced:
		name+='_bal'
	plt.savefig(name+'.png', bbox_inches='tight')
	acc, prec, recall, f1 = data.calculate_class_metrics(cm)

	return acc, prec, recall, f1

def compute_window_vals(series):
	features = []
	for s in series:
		s = np.array(s)
		mean = np.mean(s, axis=1)
		std = np.std(s, axis=1)
		slopes = get_slopes(s)
		f = []
		for i in range(len(mean)):
			f.append(mean[i])
			f.append(std[i])
			f.append(slopes[i])
		features.append(f)
	return np.array(features)


def get_slopes(series):
	slopes = []
	x = np.array(range(len(series)))
	for r in series.T:
		slope, inter = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T,r, rcond=None)[0]
		slopes.append(slope)

	return slopes
	






