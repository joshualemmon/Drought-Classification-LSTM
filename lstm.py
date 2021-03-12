import torch
import torch.nn as nn
import torch.utils.data as td
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random

class LSTM(nn.Module):
	def __init__(self, input_size=18, hidden_layer_size=100, output_size=6, num_layers=2):
		super().__init__()

		self.hidden_layer_size = hidden_layer_size
		self.input_size = input_size
		self.output_size = output_size
		self.num_layers = num_layers

		self.lstm = nn.LSTM(self.input_size, self.hidden_layer_size, self.num_layers)

		self.hidden_cell(torch.zeros(1, 1, self.hidden_layer_size), torch.zeros(1, 1, self.hidden_layer_size))

		self.linear = nn.Linear(self.hidden_layer_size, self.output_size)

		self.softmax = nn.Softmax(dim=0)

	def forward(self, input_sequence):
		lstm_output, hidden_state = self.lstm(input_sequence)
		predictions = self.softmax(self.linear(lstm_out))

		return torch.argmax(predictions, dim=1)

def init_lstm(input_size=18, hidden_layer_size=100, output_size=6, num_layers=2):
	lstm = LSTM(input_size, hidden_layer_size, output_size, num_layers)
	lstm.to(device)
	return lstm

def train_lstm(lstm, train, val, device, epochs=100, balanced=False):
	loss_func = nn.CrossEntropyLoss()
	optim = torch.optim.Adam()
	if balanced:
		train_sequences = zip(train[0], train[1])
	else:
		train_sequences = get_sequences(train)

	dl = DataLoader(train_sequences, batch_size=32)
	val_metrics = []
	loss_vals = []
	for e in range(epochs):
		batch = random.sample(train_sequences, 32)
		lstm.train(True)
		seqs, labels = zip(*batch)
		seqs, labels = seqs.to(device), labels.to(device) 
		optim.zero_grad()
		out = lstm(seqs)
		loss = loss_func(out, labels)
		loss_vals.append(loss)
		loss.backward()
		optim.step()
		lstm.train(False)
		val_metrics.append(test_lstm(lstm, val, device))
		if e%5 == 0:
			print(val_metrics[-1])

	return lstm, val_metrics, loss_vals

def test_lstm(lstm, test, device):
	labels = test['score'].dropna().tolist()
	test_sequence = get_sequences(test)
	dl = td.DataLoader(test_sequence)
	preds = []
	for inputs, label in dl:
		inputs, label = inputs.to(device), inputs.to(device)
		pred = lstm(inputs)
		preds.append(pred)

	cm = confusion_matrix(labels, pred, labels=[0,1,2,3,4,5])
	acc, prec, recall, f1 = data.calculate_metrics(cm)

	return [acc, prec, recall, f1]


def get_sequences(df):
	seqs = []
	labels = df.pop('score').dropna().astype(int)

	for i, v in enumerate(data.get_series(df)):
		seqs.append((v, labels[i]))

	return seqs

