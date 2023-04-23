import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, datasets
from torchvision.models import resnet50
from torchmetrics import Accuracy

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from functions.dataset import *
# import functions.dataset as d

batch_size = 256

data_train_dev = CustomDTS("datasets/ECG200_TRAIN.txt")
mean = data_train_dev.__getitem__(5)[0].mean()
std = data_train_dev.__getitem__(5)[0].std()

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean, std) 
])

data_train_dev = CustomDTS("datasets/ECG200_TRAIN.txt", transform=transform)
data_test = CustomDTS("datasets/ECG200_TEST.txt", transform=transform)
data_train, data_dev = torch.utils.data.random_split(data_train_dev, [0.7, 0.3], generator=torch.Generator().manual_seed(42))
train_dl = DataLoader(data_train, batch_size=batch_size, shuffle=True)
dev_dl = DataLoader(data_dev, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(data_test, batch_size=batch_size, shuffle=True)

epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
