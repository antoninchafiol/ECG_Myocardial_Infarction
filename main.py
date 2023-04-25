import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, datasets
from torchvision.models import resnet50
from torchmetrics import Accuracy
from torch.utils.data import DataLoader

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from functions.dataset import *
from functions.model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Resnet version
model = resnet50(weights=None)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model.fc.in_features
model.fc= nn.Linear(in_features=num_ftrs, out_features=2, bias=False)
model.to(device)

batch_size = 10 

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
train_eval_dl = {'train': train_dl, 'dev': dev_dl}

epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
metric = Accuracy(task="binary").to(device)

model, best_wts, last_epoch_wts, losses, accuracies = train_dev_model(model, train_eval_dl, criterion, optimizer, device, metric, scheduler=None, epochs=epochs)