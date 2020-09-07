import torch
from torch.utils.data import Dataset
import reader
from os import listdir
from os.path import join
import pandas as pd
from torchvision import transforms
import numpy as np
import torchvision.models as models

transform = transforms.Compose([transforms.ToTensor()])

def video2tensor_list(video):
    tensor_list = [transform(image_slice) for image_slice in video]
    return tensor_list

class dataloader(Dataset):
    def __init__(self, mode):
        super(dataloader, self).__init__()
        self.videoPath = '../hw4_data/TrimmedVideos/video/{}/'.format(mode)        
        self.labelPath = '../hw4_data/TrimmedVideos/label/gt_{}.csv'.format(mode)
        self.videoList = reader.getVideoList(self.labelPath)
        self.resnet50 = models.resnet50(pretrained = True, progress = True)

    def __len__(self):
        return len(self.videoList['Video_index'])

    def __getitem__(self, index):
        # video.shape: (T, H, W, 3)
        video = reader.readShortVideo(self.videoPath, self.videoList['Video_category'][index], self.videoList['Video_name'][index])
        tensor_list = video2tensor_list(video)
        with torch.no_grad():
            feature = [self.resnet50(tensor.unsqueeze(0)).squeeze(0) for tensor in tensor_list]
        label = torch.tensor(int(self.videoList['Action_labels'][index]))
        return feature, label
