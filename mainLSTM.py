
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
from numpy.random import randint, ranf
import matplotlib.pyplot as plt
from threading import *

from functions.dataset import *
from functions.model import *
from functions.training import *
# ---------- END Imports ----------

def randomizeParams(t_nb):
    result = []
    for i in range(t_nb):
        result.append(
            {
                "epoch": randint(10,250), 
                "batch_size": randint(10,50), 
                "optim_lr": (ranf() + 0.015) % 0.01, 
                'train_dev_split': (ranf() + 0.2) % 1,
                'lstm_input_s':96, # X length = 96
                'lstm_output_s':1,
                'lstm_hidden_s':256,
                'lstm_layer_dim': 1,
                "split_seed": 42,
                'batch_first':True,
                "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            }
        )
    return result

def main(params):
    model = LSTModel(params['lstm_input_s'], params['lstm_hidden_s'], params['lstm_output_s'], params['lstm_hidden_s'])
    model.to(params['device'])
        
    data_train = lstmDts("datasets/ECG200_TRAIN.txt")
    data_test = lstmDts("datasets/ECG200_TEST.txt")
    # testi = 0 
    # x,y = data_train.__getitem__(0)
    # print(x)
    # print(y)
    data_train, data_dev = torch.utils.data.random_split(data_train, [params["train_dev_split"], 1-params["train_dev_split"]], generator=torch.Generator().manual_seed(params['split_seed']))

    data_ld = {
        "train": DataLoader(data_train, batch_size=params['batch_size'], shuffle=True),
        "dev": DataLoader(data_dev, batch_size=params['batch_size'], shuffle=False),
        "test": DataLoader(data_test, batch_size=params['batch_size'], shuffle=True)
    }

    loss_fn = nn.CrossEntropyLoss()
    optimiz = optim.Adam(model.parameters(), lr=params["optim_lr"])
    metric = F1Score(task="binary").to(params["device"])

    # for (X,Y) in data_ld["test"]:
    #     print(X.shape) 
    #     print(Y.shape) 
    model, best_wts, last_epoch_wts, losses, accuracies = train_dev_modelLSTM(
        model, 
        {"train": data_ld["train"] , "dev":data_ld["dev"]},
        loss_fn, 
        optimiz, 
        params["device"], 
        metric, 
        epochs=params["epoch"]
    )



params = {
    "epoch": 10, 
    "batch_size": 10, 
    "optim_lr": 0.001, 
    'train_dev_split': 0.6,
    "split_seed": 42,
    'lstm_input_s':1, 
    'lstm_hidden_s':256,
    'lstm_output_s':1,
    'lstm_layer_dim': 1,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

if __name__ == '__main__':
    # thread_nb = 5
    # r_params = randomizeParams(thread_nb)
    # print(r_params[0])
    # threads = []
    # for i in range(thread_nb):
    #     x = Thread(target=main, args=(r_params[i],))
    #     threads.append(x)
    #     x.start()

    # for thread in threads:
    #     thread.join()
    main(params)


