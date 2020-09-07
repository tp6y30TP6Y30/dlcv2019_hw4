import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class RNN(nn.Module):
    def __init__(self, feature_size = 1000, hidden_size = 1000, output_size = 11):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size = feature_size, hidden_size = hidden_size, num_layers = 2, batch_first = True, bidirectional = True)
        self.fc = nn.Sequential(
                        nn.Linear(feature_size + hidden_size, feature_size),
                        nn.ReLU(True),
                        nn.Linear(feature_size, output_size),
                        nn.Softmax(dim = 2),
                  )

    def forward(self, input, length_list): 
        input_pack = rnn_utils.pack_padded_sequence(input, length_list, batch_first = True)
        output, _ = self.rnn(input_pack)
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first = True)
        output = self.fc(output)
        pred = torch.mean(output, dim = 1, keepdim = True)
        return pred

