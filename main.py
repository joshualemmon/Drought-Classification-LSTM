import pandas as pd
import numpy as np
import data
import lstm
import forest
import sklearn as sk
import torch
import argparse as ap


def main(args):
	save = args.save if args.save else False
	preprocess = args.preprocess if args.preprocess else False
	series_length = args.series_length if args.series_length else 7
	train_path = args.train_path if args.train_path else 'data/train.csv'
	val_path = args.val_path if args.val_path else 'data/validation.csv'
	test_path = args.test_path if args.test_path else 'data/test.csv'
	print('Reading data')
	read_data.read_data(train_path,val_path,test_path, preprocess, save, series_length)


if __name__ == '__main__':
	parser = ap.ArgumentParser()
	parser.add_argument('--save', '-s', default=False, action='store_true')
	parser.add_argument('--preprocess', '-p', default=False, action='store_true')
	parser.add_argument('--series_length', '-sl', type=int)
	parser.add_argument('--train_path', '-trp', type=str)
	parser.add_argument('--test_path', '-tp', type=str)
	parser.add_argument('--val_path', '-vp', type=str)

	main(parser.parse_args())	