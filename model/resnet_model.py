import torch.nn as nn
from torchvision.models import resnet50


class ResOWModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(pretrained=True)
        for p in self.resnet.parameters():
            p.requires_grad = False

        n_ft = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(n_ft, 28)

        
    def forward(self, x):
        return self.resnet(x)
