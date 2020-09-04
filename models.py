import torchvision.models as models
import torch
import torch.nn as nn

def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

class Extractor(nn.Module):
    def __init__(self, resnet_output = 1000):
        super(Extractor, self).__init__()
        self.resnet50 = models.resnet50(pretrained = True, progress = True)
        self.fc = nn.Sequential(
                        nn.Linear(resnet_output, 2048),
                        nn.ReLU(True),
                        nn.Linear(2048, 1024),
                        nn.ReLU(True),
                        nn.Linear(1024, 11),
                        nn.Softmax(dim = 1),
                  )
        weights_init_uniform(self.fc)

    def forward(self, input):
        input = input.squeeze(0)
        feature = self.resnet50(input)
        predict = self.fc(feature)
        return predict

