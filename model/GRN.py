import torch.nn as nn
from model.Gate import Gate


class GRN(nn.Module):
    def __init__(self, inp_dim_x, inp_dim_m, inter_dim, out_dim, bias=True, gate_bias=True, dual=False):
        super(GRN, self).__init__()
        # print(f"{inp_dim_x}, {inp_dim_m}, {inter_dim}, {out_dim}, {bias}, {gate_bias}, {dual}")
        self.dual = dual
        self.w1 = nn.Linear(in_features=inter_dim, out_features=out_dim, bias=bias)
        self.w2 = nn.Linear(in_features=inp_dim_x, out_features=inter_dim, bias=bias)
        if dual:
            self.w3 = nn.Linear(in_features=inp_dim_m, out_features=inter_dim, bias=False)
        self.activation = nn.ELU()
        self.gate = Gate(dim=out_dim, bias=gate_bias)
        self.layer_norm = nn.LayerNorm(out_dim, eps=1e-6)
        self.projection = nn.Linear(in_features=inp_dim_x, out_features=out_dim, bias=False)

    def forward(self, x):
        if self.dual:
            y = self.w2(x[0]) + self.w3(x[1])
        else:
            y = self.w2(x)

        y = self.activation(y)
        y = self.w1(y)
        y = self.gate(y)

        # residual connection is based on the primal vector
        projected_x = self.projection(x[0]) if self.dual else self.projection(x)

        return self.layer_norm(projected_x + y)




