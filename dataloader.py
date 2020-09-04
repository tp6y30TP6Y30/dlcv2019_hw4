import torch
from torch.utils.data import Dataset
import reader
from os import listdir
from os.path import join
import pandas as pd
from torchvision import transforms
import numpy as np

task2str = {1: "TrimmedVideos", 2: "TrimmedVideos", 3: "FullLengthVideos"}

transform = transforms.Compose([transforms.ToTensor()])

def video2tensor(video):
    tensor_list = [transform(image_slice) for image_slice in video]
    video_tensor = torch.stack((tensor_list), dim = 0)
    return video_tensor

class dataloader(Dataset):
    def __init__(self, task, mode):
        super(dataloader, self).__init__()
        self.videoPath = 'hw4_data/{}/video/{}/'.format(task2str[task], mode)        
        self.labelPath = 'hw4_data/{}/label/gt_{}.csv'.format(task2str[task], mode)
        self.videoList = reader.getVideoList(self.labelPath)

    def __len__(self):
        return len(self.videoList['Video_index'])

    def __getitem__(self, index):
        # video.shape: (T, H, W, 3)
        video = reader.readShortVideo(self.videoPath, self.videoList['Video_category'][index], self.videoList['Video_name'][index])
        video_tensor = video2tensor(video)
        label = np.array([self.videoList['Action_labels'][index]] * video_tensor.size(0)).astype(np.int64)
        return video_tensor, label
