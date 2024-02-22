import torch.nn as nn


class Gate(nn.Module):
    def __init__(self, dim, bias=True):
        super(Gate, self).__init__()
        self.w1 = nn.Linear(in_features=dim, out_features=dim, bias=bias)
        self.w2 = nn.Linear(in_features=dim, out_features=dim, bias=bias)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        mask = self.activation(self.w1(x))
        out = self.w2(x)
        return mask * out
