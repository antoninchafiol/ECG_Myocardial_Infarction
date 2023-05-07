
# ---------- START Imports ----------
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms, datasets
from torchmetrics import Accuracy, F1Score
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from functions.dataset import *
from functions.model import *
# ---------- END Imports ----------

params = {
    "epoch": 100, 
    "batch_size": 10, 
    "optim_lr": 0.001, 
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'train_dev_split': 0.6,
    "split_seed": 42
}

model = nn.LSTM()
model.to(params['device'])

data_train = CustomDTS("datasets/ECG200_TRAIN.txt")
nb = np.random.randint(len(data_train))
mean = data_train.__getitem__(nb)[0].mean()
std = data_train.__getitem__(nb)[0].std()

transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean, std) 
])

data_train = CustomDTS("datasets/ECG200_TRAIN.txt", transform=transform)
data_test = CustomDTS("datasets/ECG200_TEST.txt", transform=transform)
data_train, data_dev = torch.utils.data.random_split(data_train, [params["train_dev_split"], 1-params["train_dev_split"]], generator=torch.Generator().manual_seed(params['split_seed']))

data_ld = {
    "train": DataLoader(data_train, batch_size=params['batch_size'], shuffle=True),
    "dev": DataLoader(data_dev, batch_size=params['batch_size'], shuffle=False),
    "test": DataLoader(data_test, batch_size=params['batch_size'], shuffle=True)
}

loss_fn = nn.CrossEntropyLoss()
optimiz = optim.Adam(model.parameters(), lr=params["optim_lr"])
metric = F1Score(task="binary").to(params["device"])


model, best_wts, last_epoch_wts, losses, accuracies = train_dev_model(
    model, 
    {"train": data_ld["train"] , "dev":data_ld["dev"]},
    loss_fn, 
    optimizer, 
    params["device"], 
    metric, 
    epochs=params["epoch"]
)


losses = torch.tensor(losses).cpu()
accuracies = torch.tensor(accuracies).cpu()


plt.plot([i for i in range(params["epoch"])], losses, label='Loss')
plt.show()
plt.plot([i for i in range(params["epoch"])], accuracies, label='Accuracy')
plt.show()


model, accuracy_test = test(model, data_ld["test"], metric, params["device"]) 

torch.save(model.state_dict(), "weights/RunMainLSTM.pth")