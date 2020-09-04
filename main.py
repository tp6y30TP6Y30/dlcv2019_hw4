import argparse
from dataloader import dataloader
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from models import Extractor
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type = int, default = 0)
    parser.add_argument('--mode', type = str)
    parser.add_argument('--load', type = int, default = -1)
    return parser.parse_args()

def freeze_resnet50(model):
    for param in model.resnet50.parameters():
        param.requires_grad = False

def run(args):
    torch.multiprocessing.freeze_support()
    EPOCH = 100

    train_dataloader = dataloader(args.task, 'train')
    train_data = DataLoader(train_dataloader, batch_size = 1, shuffle = True, num_workers = 6, pin_memory = True)

    test_dataloader = dataloader(args.task, 'valid')
    test_data = DataLoader(test_dataloader, batch_size = 1, shuffle = False, num_workers = 6, pin_memory = True)

    model = Extractor()
    # freeze_resnet50(model)
    model.cuda().float()

    optimizer = torch.optim.Adam(filter(lambda param : param.requires_grad, model.parameters()), lr = 1e-5)
    
    loss_func = nn.CrossEntropyLoss()
    loss_func.cuda().float()

    for epoch in range(args.load + 1, EPOCH):
        total_loss = 0
        model.train()
        for index, (video, label) in enumerate(tqdm(train_data, ncols = 80, desc = '[Training] epoch ({} / {})'.format(epoch, EPOCH))):
            batch_video, batch_label = video.to(device), label.to(device)
            batch_label = batch_label.squeeze(0)
            prediction = model(batch_video)

            loss = loss_func(prediction, batch_label)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print('epoch: {} / {}\ntrain_avg_loss: {:.4f}'.format(epoch, EPOCH, total_loss / len(train_data)))

        total_loss = 0
        model.eval()
        with torch.no_grad():
            for index, (video, label) in enumerate(tqdm(test_data, ncols = 80, desc = '[Testing] epoch ({} / {})'.format(epoch, EPOCH))):
                batch_video, batch_label = video.to(device), label.to(device)
                batch_label = batch_label.squeeze(0)
                prediction = model(batch_video)
                loss = loss_func(prediction, batch_label)
                total_loss += loss
            print('epoch: {} / {}\ntest_avg_loss: {:.4f}'.format(epoch, EPOCH, total_loss / len(test_data)))

if __name__ == '__main__':
    args = _parse_args()
    run(args)