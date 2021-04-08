import torch
import torch.nn as nn
import torch.utils.data as td
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
import random
import data
import matplotlib.pyplot as plt
import seaborn as sns


class LSTM(nn.Module):
	def __init__(self, input_size=18, hidden_layer_size=512, output_size=6, num_layers=2, series_length=7, device='cpu'):
		super().__init__()

		self.hidden_layer_size = hidden_layer_size
		self.input_size = input_size
		self.output_size = output_size
		self.num_layers = num_layers
		self.series_length = series_length
		self.device = device

		self.lstm = nn.LSTM(self.input_size, self.hidden_layer_size, self.num_layers, dropout=0.1)

		self.hidden_state = (torch.zeros(self.num_layers, self.series_length, self.hidden_layer_size).to(self.device), \
			                 torch.zeros(self.num_layers, self.series_length, self.hidden_layer_size).to(self.device))

		self.linear = nn.Linear(self.hidden_layer_size, self.output_size)


	def forward(self, input_sequence):
		self.hidden_state = (torch.zeros(self.num_layers, self.series_length, self.hidden_layer_size).to(self.device), \
			                 torch.zeros(self.num_layers, self.series_length, self.hidden_layer_size).to(self.device))
		lstm_output, self.hidden_state = self.lstm(input_sequence, self.hidden_state)
		linear_out = self.linear(lstm_output[:,-1,:])

		return linear_out

def init_lstm(input_size=18, hidden_layer_size=512, output_size=6, num_layers=2, series_length=7, device='cpu'):
	lstm = LSTM(input_size, hidden_layer_size, output_size, num_layers, series_length, device)
	lstm.to(device)
	return lstm

def train_lstm(lstm, train, val, device, batch_size=64, epochs=5000, balanced=False):
	loss_func = nn.CrossEntropyLoss()
	optim = torch.optim.Adam(lstm.parameters())
	if balanced:
		train_x = train[0]
		train_y = train[1]
		train_sequences = []
		for i in range(len(train_x)):
			train_sequences.append((train_x[i], train_y[i]))
	else:
		train_sequences = get_sequences(train)
	val_sequences = get_sequences(val)
	val_set = random.sample(val_sequences, 10000)
	val_metrics = []
	loss_vals = []
	for e in range(epochs):
		print(f'Epoch {e}')
		batch = random.sample(train_sequences, batch_size)
		lstm.train(True)
		seqs, labels = zip(*batch)
		seqs, labels = torch.Tensor(seqs).to(device), torch.LongTensor(labels).to(device)
		
		optim.zero_grad()
		out = lstm(seqs)
		loss = loss_func(out, labels)
		loss.backward()
		optim.step()
		lstm.train(False)
		loss_vals.append(loss.item())
		val_metrics.append(test_lstm(lstm, val_set, device, seq=True, balanced=balanced))
		if e%5 == 0:
			print('Val metrics')
			print(val_metrics[-1])

	return lstm, val_metrics, loss_vals

def test_lstm(lstm, test, device, seq=False, balanced=False):
	if seq:
		test_sequence = test
	else:
		test_sequence = get_sequences(test)
	inputs, labels = zip(*test_sequence)
	preds = []
	for i in inputs:
		i = torch.Tensor(i).to(device)
		i = i.unsqueeze(0)
		pred = lstm(i)
		preds.append(torch.argmax(pred).item())
	labels = list(labels)
	cm = confusion_matrix(labels, preds, labels=[0,1,2,3,4,5])
	acc, prec, recall, f1 = data.calculate_class_metrics(cm)

	if seq == False:
		fig = plt.figure()
		disp = ConfusionMatrixDisplay(cm, display_labels=['None', 'D0', 'D1', 'D2', 'D3', 'D4'])
		disp.plot()
		if balanced:
			plt.savefig('images/lstm_cm_test_balanced.png', bbox_inches='tight')
		else:
			plt.savefig('images/lstm_cm_test.png', bbox_inches='tight')


	return list(data.calculate_average_metrics(acc, prec, recall, f1))


def get_sequences(df):
	seqs = []
	labels = df.pop('score').dropna().astype(int).tolist()

	for i, v in enumerate(data.get_series(df)):
		seqs.append((v, labels[i]))

	return seqs

def plot_lstm_values(val_metrics, loss, bal=False):
	sns.set_theme()
	val_metrics = np.array(val_metrics)

	val_acc = val_metrics[:,0]
	val_prec = val_metrics[:,1]
	val_recall = val_metrics[:,2]
	val_f1 = val_metrics[:,3]

	fig = plt.figure()
	plt.plot(val_acc)
	plt.title('LSTM Validation Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	if bal:
		plt.savefig('images/validation_acc_plot_lstm_bal.png', bbox_inches='tight')
	else:
		plt.savefig('images/validation_acc_plot_lstm.png', bbox_inches='tight')
	fig.clf()

	fig = plt.figure()
	plt.plot(val_prec)
	plt.title('LSTM Validation Precision')
	plt.ylabel('Precision')
	plt.xlabel('Epochs')
	if bal:
		plt.savefig('images/validation_prec_plot_lstm_bal.png', bbox_inches='tight')
	else:
		plt.savefig('images/validation_prec_plot_lstm.png', bbox_inches='tight')
	fig.clf()

	fig = plt.figure()
	plt.plot(val_recall)
	plt.title('LSTM Validation Recall')
	plt.ylabel('Recall')
	plt.xlabel('Epochs')
	if bal:
		plt.savefig('images/validation_recall_plot_lstm_bal.png', bbox_inches='tight')
	else:
		plt.savefig('images/validation_recall_plot_lstm.png', bbox_inches='tight')
	fig.clf()

	fig = plt.figure()
	plt.plot(val_f1)
	plt.title('LSTM Validation F1 Score')
	plt.ylabel('F1 Score')
	plt.xlabel('Epochs')
	if bal:
		plt.savefig('images/validation_f1_plot_lstm_bal.png', bbox_inches='tight')
	else:
		plt.savefig('images/validation_f1_plot_lstm.png', bbox_inches='tight')
	fig.clf()

	fig = plt.figure()
	plt.plot(loss)
	plt.title('LSTM Training Loss')
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	if bal:
		plt.savefig('images/loss_plot_lstm_bal.png', bbox_inches='tight')
	else:
		plt.savefig('images/loss_plot_lstm.png', bbox_inches='tight')