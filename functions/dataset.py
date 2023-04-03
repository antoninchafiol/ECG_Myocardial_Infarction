import numpy as np
from torch.utils.data import Dataset

class CustomDTS(Dataset):
    def __init__(self, file, transform=None):
        X = []
        Y = []
        for line in open(file,'r').readlines():
            Y.append([float(i) for i in line.split()][0])
            X.append([float(i) for i in line.split()][1:])
        X = np.array(X, dtype='float32')
        Y = np.array(Y, dtype='int8')
        self.X = X
        self.Y = Y
        self.transform=transform
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        return self.X[index],self.Y[index]