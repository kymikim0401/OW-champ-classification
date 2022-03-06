import torch.nn as nn


class OWmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=1),
        
            nn.Conv2d(8, 20, kernel_size=3, stride=1),
            nn.BatchNorm2d(20),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=4, stride=1) #expected dimension = 20x247x247
        )
        
        self.linear=nn.Sequential(
            nn.Linear(20*248*248, 28),
            nn.Softmax()
        )
        
    def forward(self, x):
        y = self.cnn(x)
        y = y.view(y.size(0), -1)
        z = self.linear(y)
        
        
        return z 