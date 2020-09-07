import argparse
from dataloader import dataloader
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from models import RNN
import torch
import torch.nn as nn
import os
import torch.nn.utils.rnn as rnn_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', type = str)
	parser.add_argument('--load', type = int, default = -1)
	return parser.parse_args()

def freeze_resnet50(model):
	for param in model.resnet50.parameters():
		param.requires_grad = False

def collate_fn(data):
	data = list(map(list, data))
	data.sort(key = lambda x : x[1], reverse = True)
	video, frame_size, label = zip(*data)
	video, frame_size, label = list(video), list(frame_size), list(label)
	video = rnn_utils.pad_sequence(video, batch_first = True, padding_value = 0)
	frame_size = torch.stack(frame_size, dim = 0)
	label = torch.stack(label, dim = 0)
	return video, frame_size, label

def cal_accuracy(batch_pred, batch_label):
	hit, total = torch.sum(torch.argmax(batch_pred) == batch_label).item(), batch_pred.size(0)
	return hit, total

def run(args):
	torch.multiprocessing.freeze_support()
	EPOCH = 30
	batch_size = 3
	train_dataloader = dataloader('train')
	train_data = DataLoader(train_dataloader, batch_size = batch_size, shuffle = True, num_workers = 0, pin_memory = True, collate_fn = collate_fn)

	test_dataloader = dataloader('valid')
	test_data = DataLoader(test_dataloader, batch_size = batch_size, shuffle = False, num_workers = 0, pin_memory = True, collate_fn = collate_fn)

	model = RNN()
	if args.load != -1:
		checkpoint = torch.load('models/model_epoch{}.pkl'.format(args.load))
		model.rnn.load_state_dict(checkpoint['model_rnn'])
		model.fc.load_state_dict(checkpoint['model_fc'])

	freeze_resnet50(model)
	model.cuda().float()
	optimizer = torch.optim.Adam(filter(lambda param : param.requires_grad, model.parameters()), lr = 1e-6, weight_decay = 0.012)

	loss_func = nn.CrossEntropyLoss(reduction = 'mean')
	loss_func.cuda().float()

	with open('loss_accuracy.txt', 'w' if args.load == -1 else 'a') as f:
		pass

	for epoch in range(args.load + 1, EPOCH):
		total_loss = 0
		model.train()
		all_hit = all_total = 0
		for index, (video, frame_size, label) in enumerate(tqdm(train_data, ncols = 80, desc = '[Training] epoch ({} / {})'.format(epoch, EPOCH))):
			batch_video, batch_frame_size, batch_label = video.to(device), frame_size.to(device), label.to(device)
			prediction = model(batch_video, batch_frame_size)
			prediction = prediction.squeeze(1)
			hit, total = cal_accuracy(prediction, batch_label)
			all_hit += hit
			all_total += total
			loss = loss_func(prediction, batch_label)
			total_loss += loss
			loss.backward()
			optimizer.step()
			
		print('epoch: {} / {}\ntrain_avg_loss: {:.4f}'.format(epoch, EPOCH, total_loss / len(train_data)))
		print('train_accuracy: {:.1f}%'.format(all_hit / all_total * 100))
		with open('loss_accuracy.txt', 'a') as f:
			f.write('epoch: {} / {}\ntrain_avg_loss: {:.4f}\ttrain_accuracy: {:.1f}%\n'.format(epoch, EPOCH, total_loss / len(train_data), all_hit / all_total * 100))

		total_loss = 0
		model.eval()
		all_hit = all_total = 0
		with torch.no_grad():
			for index, (video, label, length_list) in enumerate(tqdm(test_data, ncols = 80, desc = '[Testing] epoch ({} / {})'.format(epoch, EPOCH))):
				batch_video, batch_label, batch_length_list = video.to(device), label.to(device), length_list
				prediction = model(batch_video, batch_length_list)
				prediction = prediction.squeeze(1)
				hit, total = cal_accuracy(prediction, batch_label)
				all_hit += hit
				all_total += total
				loss = loss_func(prediction, batch_label)
				total_loss += loss

			print('epoch: {} / {}\ntest_avg_loss: {:.4f}'.format(epoch, EPOCH, total_loss / len(test_data)))
			print('test_accuracy: {:.1f}%'.format(all_hit / all_total * 100))
			with open('loss_accuracy.txt', 'a') as f:
				f.write('test_avg_loss: {:.4f}\ttest_accuracy: {:.1f}%\n'.format(total_loss / len(test_data), all_hit / all_total * 100))

		if not os.path.exists('models/'):
			os.mkdir('models/')
		torch.save({'model_rnn': model.rnn.state_dict(),
					'model_fc': model.fc.state_dict(),
				   }, 'models/model_epoch{}.pkl'.format(epoch))
			
if __name__ == '__main__':
	args = _parse_args()
	run(args)