from torch.utils.data.dataset import Dataset
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np


class OverWatchDataset(Dataset):
    def __init__(self):
        self.transforms = transforms.Compose([
             transforms.PILToTensor(),
             transforms.ConvertImageDtype(torch.float),
        ])
        df = pd.read_csv('Data.csv')
        self.fpaths = df['file']
        self.labels = pd.get_dummies(df['class']).values 
        
    def __getitem__(self, index):
        fp = self.fpaths[index]
        img = Image.open(fp)
        data = self.transforms(img)
        label = torch.from_numpy(self.labels[index].astype(np.float32))
        return (data, label)

    def __len__(self):
        return len(self.fpaths)