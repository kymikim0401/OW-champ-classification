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
        #Line 11-12: Fine-tuning Resnet 50 models / 28 = number of class (= number of champs in OW)
        
    def forward(self, x):
        x = self.resnet(x)
        return x
