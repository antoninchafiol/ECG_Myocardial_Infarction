import numpy as np
from torch.utils.data import Dataset

class CustomDTS(Dataset):
    def __init__(self, file, transform=None):
        X = []
        Y = []
        for line in open(file,'r').readlines():
            tmp = ([float(i) for i in line.split()][0])
            if tmp==-1.0:
                Y.append(0.0)
            else:
                Y.append(tmp)
            X.append([float(i) for i in line.split()][1:])
        X = np.array(X, dtype='float32')
        X = X.reshape((X.shape[0], X.shape[1],1,1))
        Y = np.array(Y, dtype='int8')
        self.X = X
        self.Y = Y
        self.transform=transform
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        if self.transform is not None:
            X = self.transform(X)
        return X,Y
    
class lstmDts(Dataset):
    def __init__(self, file, transform=None):
        X = []
        Y = []
        for line in open(file,'r').readlines():
            tmp = ([float(i) for i in line.split()][0])
            if tmp==-1.0:
                Y.append(0.0)
            else:
                Y.append(tmp)
            X.append([float(i) for i in line.split()][1:])

        X = np.array(X, dtype='float32')
        Y = np.array(Y, dtype='float32')
        # X = X.reshape(X.shape[0],X.shape[1])
        # print(type(X))
        self.X = X
        self.Y = Y
        self.transform=transform

    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        if self.transform is not None:
            X = self.transform(X)
        return X,Y
    

class ECGResNet(Dataset):
    def __init__(self, file, transform=None):
        X = []
        Y = []
        for line in open(file,'r').readlines():
            tmp = ([float(i) for i in line.split()][0])
            if tmp==-1.0:
                Y.append(0.0)
            else:
                Y.append(tmp)
            X.append([float(i) for i in line.split()][1:])
        X = np.array(X, dtype='float32')
        X = X.reshape((X.shape[0], X.shape[1],1,1))
        Y = np.array(Y, dtype='int8')
        self.X = X
        self.Y = Y
        self.transform=transform
    def __len__(self):
        return len(self.Y)
    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        if self.transform is not None:
            X = self.transform(X)
        return X,Y