import torchvision.models as models
import torch
import torch.nn as nn

class Extractor(nn.Module):
    def __init__(self, feature_size = 1000, hidden_size = 1000, output_size = 11):
        super(Extractor, self).__init__()
        # resnet50_output_size: (T, 1000)
        self.resnet50 = models.resnet50(pretrained = True, progress = True)
        self.hidden_size = hidden_size
        self.hidden_init = self.init_hidden()
        self.level1 = nn.Sequential(
                            nn.Linear(feature_size + hidden_size, hidden_size),
                            nn.ReLU(True),
                      )
        self.level2 = nn.Sequential(
                            nn.Linear(feature_size + hidden_size, hidden_size),
                            nn.ReLU(True),
                      )
        self.level3 = nn.Sequential(
                            nn.Linear(feature_size + hidden_size, hidden_size),
                            nn.ReLU(True),
                      )
        self.predict = nn.Sequential(
                            nn.Linear(hidden_size, output_size),
                            nn.Softmax(dim = 1),
                      )
        self.weights_init_uniform(self.level1)
        self.weights_init_uniform(self.level2)
        self.weights_init_uniform(self.level3)
        self.weights_init_uniform(self.predict)

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size).cuda()

    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # apply a uniform distribution to the weights and a bias=0
            m.weight.data.uniform_(0.0, 0.5)
            m.bias.data.fill_(0)

    def sample_seperate(self, feature):
        return feature[0].unsqueeze(0), feature[int(feature.size(0) / 2)].unsqueeze(0), feature[-1].unsqueeze(0)

    def combine(self, tensor1, tensor2):
        return torch.cat((tensor1, tensor2), dim = 1)

    def forward(self, input):
        input = input.squeeze(0)
        feature = self.resnet50(input)
        feature_front, feature_mid, feature_tail = self.sample_seperate(feature)
        feature = self.level1(self.combine(feature_front, self.hidden_init))
        feature = self.level2(self.combine(feature_mid, feature))
        feature = self.level3(self.combine(feature_tail, feature))
        output = self.predict(feature)
        return output

