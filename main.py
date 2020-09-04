import argparse
from dataloader import dataloader
from torch.utils.data import DataLoader
import numpy as np
import torchvision.models as models

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type = int, default = 0)
    parser.add_argument('--mode', type = str)
    return parser.parse_args()

def run(args):
    resnet50 = models.resnet50(pretrained = True, progress = True)
    data_loader = dataloader(args.task, args.mode)
    data = DataLoader(data_loader, batch_size = 1, shuffle = True, num_workers = 0)
    for video, label in data:
        for index in range(len(video)):
            pred = resnet50(video[index])

if __name__ == '__main__':
    args = _parse_args()
    run(args)