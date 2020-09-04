import torchvision.models as models
import torch
import torch.nn as nn

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

    def forward(self, input):
        input = input.squeeze(0)
        feature = self.resnet50(input)
        predict = self.fc(feature)
        return predict

