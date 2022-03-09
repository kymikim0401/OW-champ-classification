import numpy as np
import torch
import pandas as pd
from glob import glob

a = glob('dataset/*/*')
b = []
for e in a:
    name = e.split('\\')[1]
    b.append(name)
df = pd.DataFrame({'class': b, 'file': a}).sample(frac=1).reset_index(drop=True)
df.to_csv(r'./Data.csv', index=False)