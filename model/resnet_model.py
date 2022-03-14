import torch.nn as nn
from torchvision.models import resnet50

class ResOWModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        for p in self.resnet.parameters():
            p.requires_grad = False # for I want to implement transfer learning using resnet50

        n_ft = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(n_ft, 28)
        
    def forward(self, x):
        x = self.resnet(x)
        #x_softmax = self.softmax(x)
        return x
