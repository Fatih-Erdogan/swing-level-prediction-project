import torch
import torch.nn as nn

from model.baseline.metadata_encoder import MetadataEncoder
from model.BiLSTM import TemporalEnricherLSTM


# BASELINE MODEL
class LSTMBaseline(nn.Module):
    def __init__(self, inp_dim, metadata_dim, hid_dim, num_layers, bidirectional,
                 metadata_bias, metadata_gate_bias, dropout):
        super(LSTMBaseline, self).__init__()
        self.directional_bias_vectors = nn.Embedding(num_embeddings=2, embedding_dim=metadata_dim)
        self.metadata_encoder = MetadataEncoder(metadata_dim, metadata_dim*4, hid_dim, num_layers, bidirectional,
                                                bias=metadata_bias, gate_bias=metadata_gate_bias)
        self.lstm = TemporalEnricherLSTM(inp_dim, hid_dim, num_layers, bidirectional=bidirectional, dropout=dropout)
        out_state = hid_dim * 2 if bidirectional else hid_dim
        self.reg_head = nn.Linear(in_features=out_state, out_features=1)
        self.criterion = nn.MSELoss()
        self.value_only = True

    def forward(self, inp):
        x, metadata, label, _, _ = inp
        bias_vector = self.directional_bias_vectors(torch.div(label[:, 0].long() + 1, 2, rounding_mode='trunc'))
        init_hid_states, init_cell_states = self.metadata_encoder((metadata, bias_vector))
        x = self.lstm(x, init_hid_states, init_cell_states)
        x1 = x[:, -1, :].squeeze(1)
        x2 = x[:, 0, :].squeeze(1)
        out = self.reg_head(x1 + x2).squeeze(1)

        loss = self.criterion(out, label[:, 1])
        return out, loss

    def set_value_only(self, value_only=True):
        self.value_only = value_only
