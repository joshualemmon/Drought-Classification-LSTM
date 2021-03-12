import pandas as pd
import numpy as np
import data
import lstm
import forest
import sklearn as sk
import torch
import argparse as ap


def main(args):
	save = args.save
	preprocess = args.preprocess
	series_length = args.series_length if args.series_length else 7
	balance = args.balance
	make_rand_forest = args.rand_forest
	make_time_series_forest = args.time_series_forest
	make_lstm = args.lstm
	if preprocess:
		train_path = args.train_path if args.train_path else 'data/train.csv'
		val_path = args.val_path if args.val_path else 'data/validation.csv'
		test_path = args.test_path if args.test_path else 'data/test.csv'
	else:
		train_path = args.train_path if args.train_path else 'data/train_processed.csv'
		val_path = args.val_path if args.val_path else 'data/validation_processed.csv'
		test_path = args.test_path if args.test_path else 'data/test_processed.csv'		

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print('Reading data')
	train, val, test = data.read_data(train_path,val_path,test_path, preprocess, save, series_length)
	if make_rand_forest:
		rf = forest.init_random_forest()
	if make_time_series_forest:
		tsf = forest.init_time_series_forest()
	if make_lstm:
		lstm = lstm.init_lstm(device=device)
	if balance:
		print('Balancing training data')
		train_x, train_y = data.balance_data(train)
		if make_rand_forest:
			print('Training Random Forest')
			rf = forest.train_random_forest(rf, [train_x, train_y], val, True)
		if make_time_series_forest:
			print('Training Time Series Forest')
			tsf = forest.train_time_series_forest(tsf, [train_x, train_y], val, True)
		if make_lstm:
			print('Training LSTM')
			lstm, lstm_val, lstm_loss = lstm.train_lstm(lstm, [train_x, train_y], val, device, balanced=True)
	else:
		if make_rand_forest:
			print('Training Random Forest')
			rf = forest.train_random_forest(rf, train, val)
		if make_time_series_forest:
			print('Training Time Series Forest')
			tsf = forest.train_time_series_forest(tsf, train, val)
		if make_lstm:
			print('Training LSTM')
			lstm, lstm_val, lstm_loss = lstm.train_lstm(lstm, train, val, device)

	if make_rand_forest:
		rf_acc, rf_prec, rf_recall, rf_f1 = forest.test_random_forest(rf, test)
		rf_avg_acc, rf_avg_prec, rf_avg_recall, rf_avg_f1 = data.calculate_average_metrics(rf_acc, rf_prec, rf_recall, rf_f1)
		print('RF: ',rf_avg_acc, rf_avg_prec, rf_avg_recall, rf_avg_f1)
	if make_time_series_forest:
		tsf_acc, tsf_pref, tsf_recall, tsf_f1 = forest.test_time_series_forest(tsf, test)
		tsf_avg_acc, tsf_avg_prec, tsf_avg_recall, tsf_avg_f1 = forest.calculate_average_metrics(tsf_acc, tsf_pref, tsf_recall, tsf_f1)
		print('TSF: ', tsf_avg_acc, tsf_avg_prec, tsf_avg_recall, tsf_avg_f1)
	if make_lstm:
		lstm_metrics = lstm.test_lstm(lstm, test, device)
		data.plot_lstm_values(lstm_val, lstm_loss)
		print('LSTM: ', lstm_metrics)


if __name__ == '__main__':
	parser = ap.ArgumentParser()
	parser.add_argument('--save', '-s', default=False, action='store_true')
	parser.add_argument('--preprocess', '-p', default=False, action='store_true')
	parser.add_argument('--series_length', '-sl', type=int)
	parser.add_argument('--train_path', '-trp', type=str)
	parser.add_argument('--test_path', '-tp', type=str)
	parser.add_argument('--val_path', '-vp', type=str)
	parser.add_argument('--balance', '-b', default=False, action='store_true')
	parser.add_argument('--rand_forest', '-rf', default=False, action='store_true')
	parser.add_argument('--time_series_forest', '-tsf', default=False, action='store_true')
	parser.add_argument('--lstm', '-l', default=False, action='store_true')

	main(parser.parse_args())	