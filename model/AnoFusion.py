import torch as t
import sys
sys.path.append('./')
from GTblock import GTN
from GATGRU import *
import torch.nn as nn

class permute(nn.Module):
    def __init__(self):
        super(permute, self).__init__()
    def forward(self, x):
        return x.permute(0, 2, 1)

class Net(nn.Module):
    def __init__(self, node_num, edge_types, window_samples_num, dropout):
        super(Net, self).__init__()
        self.edge_types = edge_types
        self.num_channels = edge_types
        self.node_num = node_num
        self.window_samples_num = window_samples_num
        self.dropout = dropout
        self.GTN = GTN(edge_types=self.edge_types, num_channels=self.num_channels, num_layers=5, norm=False)
        self.GAT_GRU = GAT_GRU(self.window_samples_num, self.node_num, self.num_channels)
        self.flatten = nn.Flatten()
        self.linT = nn.Linear(self.window_samples_num, self.window_samples_num // 2)
    
        self.all = self.window_samples_num * self.node_num
        self.Dropout = nn.Dropout(0.2)
        self.lin1 = nn.Linear(self.all, 64)
        self.act1 = nn.LeakyReLU()
        self.lin2 = nn.Linear(64, 2)
        self._final_softmax = nn.Softmax(dim=1)
    
    def forward(self, X, A):
        X = self.Dropout(X)
        A = A.view((-1, self.node_num, self.node_num, self.edge_types))
        X = X.view((-1, self.node_num, X.shape[-1]))
        # GTN
        A = self.GTN(A)
        # GAT and GRU
        out_T = self.GAT_GRU(X, A)
        return out_T