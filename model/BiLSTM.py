import torch.nn as nn
from model.Gate import Gate


class TemporalEnricherLSTM(nn.Module):
    def __init__(self, inp_dim, hid_dim, num_layers, bidirectional=True, dropout=0.2):
        super(TemporalEnricherLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=inp_dim, hidden_size=hid_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(p=dropout)
        self.gate = Gate(dim=(2 * hid_dim if bidirectional else hid_dim), bias=True)
        self.projection = nn.Linear(in_features=inp_dim, out_features=(2 * hid_dim if bidirectional else hid_dim))
        self.layer_norm = nn.LayerNorm(2 * hid_dim if bidirectional else hid_dim)

    def forward(self, x, hidden_state, cell_state):
        #  x        -> batch_size * seq_len * inp_dim
        # states    -> (num_layers * num_directions) * batch_size * hid_dim

        y, _ = self.lstm(x, (hidden_state, cell_state))
        y = self.dropout(y)
        y = self.gate(y)

        return self.layer_norm(self.projection(x) + y)


