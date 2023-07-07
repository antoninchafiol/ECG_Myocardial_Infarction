import time
import copy
import torch
import numpy as np
from torch.optim import *
import torch.nn as nn
from tqdm import tqdm


from torch.autograd import Variable


class LSTModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.output = nn.Sigmoid()
    def forward(self, x):
        h_0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c_0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        # print(h_0.shape)
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1])
        out = self.output(out)
        out = out.squeeze(1)
        return out

