import torchvision.models as models
import torch
import torch.nn as nn

def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 0.5)
        m.bias.data.fill_(0)

def sample(feature):
    return torch.stack((feature[0], feature[int(feature.size(0) / 2)], feature[-1]), dim = 0).view(1, -1)

class Extractor(nn.Module):
    def __init__(self, sample_channels = 3000):
        super(Extractor, self).__init__()
        self.resnet50 = models.resnet50(pretrained = True, progress = True)
        self.fc = nn.Sequential(
                        nn.Linear(sample_channels, 11),
                        nn.Softmax(dim = 1),
                  )
        weights_init_uniform(self.fc)

    def forward(self, input):
        input = input.squeeze(0)
        feature = self.resnet50(input)
        s_feature = sample(feature)
        predict = self.fc(s_feature)
        return predict

