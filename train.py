from model.ow_model import OWmodel
from model.resnet_model import ResOWModel
from data.ow_dataset import OverWatchDataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim as optim
import torch


if __name__ == '__main__':
    
    Epoch = 20
    Batch_size = 20

    data = OverWatchDataset()
    loader = DataLoader(data, batch_size = Batch_size)
    loss_criteria = nn.CrossEntropyLoss()

    model = OWmodel().cuda()
    optimizer = optim.Adagrad(model.parameters(), lr=0.0002)

    for epoch in range(Epoch):
        loss_data = 0
        for x, y in loader:
            output = model.forward(x.cuda())
            loss = loss_criteria(output, y.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_data = loss.data
        print("Epoch %d: loss: %f" % (epoch, loss_data))
    torch.save(model.state_dict(), 'checkpoints/model.pth')
