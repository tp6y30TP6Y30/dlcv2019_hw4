import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.utils.rnn as rnn_utils
import torchvision.models as models

class RNN(nn.Module):
    def __init__(self, feature_size = 1000, hidden_size = 1000, output_size = 11):
        super(RNN, self).__init__()
        self.resnet50 = models.resnet50(pretrained = True, progress = True)
        self.rnn = nn.RNN(input_size = feature_size, hidden_size = hidden_size, num_layers = 2, batch_first = True, bidirectional = True)
        self.fc = nn.Sequential(
                        nn.Linear(feature_size + hidden_size, feature_size),
                        nn.ReLU(True),
                        nn.Linear(feature_size, output_size),
                        nn.Softmax(dim = 2),
                  )

    def forward(self, video, frame_size):
        video_size = video.size()[2:]
        video = video.view((-1, ) + video_size)
        feature = self.resnet50(video)
        feature = feature.view(frame_size.size(0), -1, feature.size(-1))
        feature_pack = rnn_utils.pack_padded_sequence(feature, frame_size, batch_first = True)
        output, _ = self.rnn(feature_pack)
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first = True)
        output = self.fc(output)
        predict = torch.mean(output, dim = 1, keepdim = True)
        return predict

