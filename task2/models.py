import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.utils.rnn as rnn_utils
import torch.nn.init as init
import torchvision.models as models

def depadding(feature, frame_size):
    # feature.shape: (B, T(padded), C)
    batch_size, frame, feature_size = feature.size()
    depadded_feature = []
    for index, f in enumerate(feature):
        depadded_feature.append(f[0:frame_size[index]])
    return depadded_feature

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

class LSTM(nn.Module):
    def __init__(self, feature_size = 2048, hidden_size = 2048, output_size = 11):
        super(LSTM, self).__init__()
        self.pretrain = models.resnet50(pretrained = True, progress = True)
        self.pretrain.fc = Identity()
        self.lstm1 = nn.LSTM(input_size = feature_size, hidden_size = hidden_size, num_layers = 2, batch_first = True, bidirectional = True, dropout = 0.5)
        self.fc = nn.Sequential(
                        nn.Linear(hidden_size * 2, hidden_size),
                        nn.ReLU(True),
                        nn.Linear(hidden_size, output_size),
                        nn.LogSoftmax(dim = 1),
                  )
        self._initial_weights()

    def _initial_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, video, frame_size):
        video_size = video.size()[2:]
        video = video.view((-1, ) + video_size)
        feature = self.pretrain(video)
        feature = feature.view(frame_size.size(0), -1, feature.size(-1))
        feature_pack = rnn_utils.pack_padded_sequence(feature, frame_size, batch_first = True)
        output, _ = self.lstm1(feature_pack)
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first = True)
        # depadded_feature = depadding(output, frame_size)
        predict = []
        for index in range(len(output)):
            predict.append(self.fc(output[index][frame_size[index] // 2].unsqueeze(0)))
        predict = torch.stack(predict, dim = 0)
        return predict

