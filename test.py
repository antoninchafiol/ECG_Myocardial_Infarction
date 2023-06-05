import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from functions.dataset import *
from functions.model import *
from functions.training import *

data_train = lstmDts("datasets/ECG200_TRAIN.txt")

for i in data_train:
    print(i)