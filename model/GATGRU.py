import torch
import torch.nn as nn
from GAT import GATNet

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class permute(nn.Module):
    def __init__(self):
        super(permute, self).__init__()
    def forward(self, x):
        return x.permute(0, 2, 1)


class GAT_GRU(nn.Module):
    def __init__(self, num_f, dim_with_c, num_channel):
        super(GAT_GRU, self).__init__()
        self.num_f = num_f
        self.in_dim = dim_with_c
        self.out_dim = dim_with_c
        self.num_channels = num_channel
        self.per1 = permute()
        self.gru = nn.GRU(input_size=self.in_dim, hidden_size=self.out_dim, num_layers=1, batch_first=True, bidirectional=False)
        self.BN = torch.nn.BatchNorm1d(self.num_f)
        self.linear = nn.Linear(self.out_dim, self.out_dim)
        self.gat_layer = GATNet(in_c=20, hid_c=self.out_dim, out_c=20, n_heads=6)
        self.weight = nn.Parameter(torch.Tensor(1, 1))
        self.soft = nn.Parameter(torch.Tensor(self.num_channels))
        self.reset_parameters()

    def norm(self, A, add=True):
        if add == False:
            A = A * ((torch.eye(A.shape[1]) == 0).type(torch.FloatTensor)).unsqueeze(0)
        else:
            A = A * ((torch.eye(A.shape[1]) == 0).type(torch.FloatTensor)).unsqueeze(0).to(device) + torch.eye(A.shape[1]).type(torch.FloatTensor).unsqueeze(0).to(device)
        deg = torch.sum(A, dim=-1)
        deg_inv = deg.pow(-1 / 2)
        deg_inv = deg_inv.view((deg_inv.shape[0], deg_inv.shape[1], 1)) * torch.eye(A.shape[1]).type(
            torch.FloatTensor).unsqueeze(0).to(device)
        A = torch.bmm(deg_inv, A)
        A = torch.bmm(A, deg_inv).float().to(device)
        return A

    def reset_parameters(self):
        nn.init.constant_(self.weight, 10)
        nn.init.constant_(self.soft, 1 / self.num_channels)

    def gcn_conv(self, X, A):
        for i in range(X.shape[-1]):
            if i == 0:
                out = torch.bmm(A, X.unsqueeze(-2)[:, :, :, i])
                out = torch.matmul(out, self.weight)
            else:
                out = torch.cat((out, torch.matmul(torch.bmm(A, X.unsqueeze(-2)[:, :, :, i]), self.weight)),dim=-1)
        return out

    def forward(self, X, A):
        for i in range(self.num_channels):
            data = {"flow_x": X, "graph": self.norm(A[:, i, :, :])}
            if i == 0:
                X_ = self.gat_layer(data)
                X_ = X_ * self.soft[i]
            else:
                X_tmp = self.gat_layer(data)
                X_tmp = X_tmp * self.soft[i]
                X_ = X_ + X_tmp

        X_ = self.per1(X_)
        X_, hn = self.gru(X_)
        hn = torch.squeeze(hn, 0)
        res =  self.linear(hn)
        return res
