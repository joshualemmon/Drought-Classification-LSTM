from sklearn import ensemble
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pyts
import pandas as pd
import data
import matplotlib.pyplot as plt

def init_random_forest(n_trees=100, criterion="gini", max_depth=None):
	rf = ensemble.RandomForestClassifier(n_trees, criterion, max_depth)
	return rf

def init_time_series_forest(n_trees=100, criterion="gini", max_depth=None):
	tsf = pyts.classification.TimeSeriesForest(n_estimators=n_trees, criterion=criterion, max_depth=max_depth)
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
	val_acc, val_prec, val_recall, val_f1 = test_random_forest(rf, val, True)
	print(f"Random Forest Trained\nVal Acc. {val_acc}\nVal Prec. {val_prec}\nVal Recall {val_recall}\nVal F1 {val_f1}")
	return rf

def train_time_series_forest(tsf, train, val, balanced=False):
	if balanced:
		labels = train[1]
		train = train[0]
	else:
		labels = train.pop('score').astype(int)
		train = get_series(train)

	tsf.fit(train, labels)
	val_acc, val_prec, val_recall, val_f1 = test_random_forest(rf, val, True)
	print(f"Time Series Forest Trained\nVal Acc. {val_acc}\nVal Prec. {val_prec}\nVal Recall {val_recall}\nVal F1 {val_f1}")
	return rf

def test_random_forest(rf, test, val=False):
	"""
	Test the trained random forest model and return calculated metrics.
	"""
	test = test[~test['score'].isnull()]
	labels = test.pop('score').astype(int)
	test = test.drop(columns=['series'])
	acc = 0
	preds = []
	for i, row in test.iterrows():
		pred = rf.predict(row)
		preds.append(int(pred))
	cm = confusion_matrix(labels, preds, labels=[0,1,2,3,4,5])
	disp = ConfusionMatrixDisplay(cm, display_labels=['None', '	D0', 'D1', 'D2', 'D3', 'D4'])
	fig = disp.figure
	if val:
		plt.savefig('images/random_forest_cm_val.png', bbox_inches='tight')
	else:
		plt.savefig('images/random_forest_cm_test.png', bbox_inches='tight')
	acc, prec, recall, f1 = data.calculate_metrics(cm)


	return acc, prec, recall, f1

def test_time_series_forest(tsf, test, val=False):
	labels = test.pop('score').dropna().tolist()
	test_series = data.get_series(test)
	preds = tsf.predict(test_series)
	cm = confusion_matrix(labels, preds, labels=[0,1,2,3,4,5])
	disp = ConfusionMatrixDisplay(cm, display_labels=['None', '	D0', 'D1', 'D2', 'D3', 'D4'])
	fig = disp.figure
	if val:
		plt.savefig('images/time_series_forest_cm_val.png', bbox_inches='tight')
	else:
		plt.savefig('images/time_series_forest_cm_test.png', bbox_inches='tight')
	acc, prec, recall, f1 = data.calculate_metrics(cm)

	return acc, prec, recall, f1



