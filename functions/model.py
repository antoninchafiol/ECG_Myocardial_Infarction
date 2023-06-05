import time
import copy
import torch
import numpy as np
from torch.optim import *
import torch.nn as nn
from tqdm import tqdm


from torch.autograd import Variable


class LSTModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_o, _) = self.lstm(x)
        out = self.fc(h_o[-1])
        return out

