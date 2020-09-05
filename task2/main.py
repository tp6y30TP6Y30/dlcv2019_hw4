import argparse
from dataloader import dataloader
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from models import Extractor
import torch
import torch.nn as nn
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	parser.add_argument('--load', type = int, default = -1)
	return parser.parse_args()

def freeze_resnet50(model):
	for param in model.resnet50.parameters():
		param.requires_grad = False

def run(args):
	torch.multiprocessing.freeze_support()
	EPOCH = 30

	train_dataloader = dataloader('train')
	train_data = DataLoader(train_dataloader, batch_size = 1, shuffle = True, num_workers = 6, pin_memory = True)

	test_dataloader = dataloader('valid')
	test_data = DataLoader(test_dataloader, batch_size = 1, shuffle = False, num_workers = 6, pin_memory = True)

	model = Extractor()
	if args.load != -1:
		model.fc.load_state_dict(torch.load('models/linear_epoch{}.pkl'.format(args.load)))
	model.cuda().float()
	freeze_resnet50(model)
	optimizer = torch.optim.Adam(filter(lambda param : param.requires_grad, model.parameters()), lr = 1e-6, weight_decay = 0.012)
	# optimizer = torch.optim.Adam(model.parameters(), lr = 1e-8, weight_decay = 0.012)

	loss_func = nn.CrossEntropyLoss()
	loss_func.cuda().float()

	with open('loss_accuracy.txt', 'w' if args.load == -1 else 'a') as f:
		pass

	for epoch in range(args.load + 1, EPOCH):
		total_loss = 0
		model.train()
		hit = 0
		for index, (video, label) in enumerate(tqdm(train_data, ncols = 80, desc = '[Training] epoch ({} / {})'.format(epoch, EPOCH))):
			batch_video, batch_label = video.to(device), label.to(device)
			prediction = model(batch_video)
			hit += (torch.argmax(prediction).item() == batch_label.item())
			loss = loss_func(prediction, batch_label)
			total_loss += loss
			loss.backward()
			optimizer.step()
			
		print('epoch: {} / {}\ntrain_avg_loss: {:.4f}'.format(epoch, EPOCH, total_loss / len(train_data)))
		print('train_accuracy: {:.1f}%'.format(hit / len(train_data) * 100))
		with open('loss_accuracy.txt', 'a') as f:
			f.write('epoch: {} / {}\ntrain_avg_loss: {:.4f}\ttrain_accuracy: {:.1f}%\n'.format(epoch, EPOCH, total_loss / len(train_data), hit / len(train_data) * 100))

		total_loss = 0
		model.eval()
		hit = 0
		with torch.no_grad():
			for index, (video, label) in enumerate(tqdm(test_data, ncols = 80, desc = '[Testing] epoch ({} / {})'.format(epoch, EPOCH))):
				batch_video, batch_label = video.to(device), label.to(device)
				prediction = model(batch_video)
				hit += (torch.argmax(prediction).item() == batch_label.item())
				loss = loss_func(prediction, batch_label)
				total_loss += loss

			print('epoch: {} / {}\ntest_avg_loss: {:.4f}'.format(epoch, EPOCH, total_loss / len(test_data)))
			print('test_accuracy: {:.1f}%'.format(hit / len(test_data) * 100))
			with open('loss_accuracy.txt', 'a') as f:
				f.write('test_avg_loss: {:.4f}\ttest_accuracy: {:.1f}%\n'.format(total_loss / len(test_data), hit / len(test_data) * 100))

		if not os.path.exists('models/'):
			os.mkdir('models/')
		torch.save(model.fc.state_dict(), 'models/linear_epoch{}.pkl'.format(epoch))
			
if __name__ == '__main__':
	args = _parse_args()
	run(args)