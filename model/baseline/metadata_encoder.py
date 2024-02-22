import torch.nn as nn
import torch
from model.GRN import GRN


# FOR BASELINE
class MetadataEncoder(nn.Module):
    def __init__(self, metadata_dim, inter_dim, out_state, num_state_layers, state_bidirectional,
                 bias=True, gate_bias=True):
        # out dim is the hid_dim of the lstm
        super(MetadataEncoder, self).__init__()
        self.lstm_hidden_encoders = nn.ModuleList()
        self.lstm_context_encoders = nn.ModuleList()
        for layer in range(num_state_layers):
            directions = 2 if state_bidirectional else 1
            for dir in range(directions):
                self.lstm_hidden_encoders.append(
                    GRN(metadata_dim, metadata_dim, inter_dim, out_state, bias, gate_bias, dual=True))
                self.lstm_context_encoders.append(
                    GRN(metadata_dim, metadata_dim, inter_dim, out_state, bias, gate_bias, dual=True))

    def forward(self, x):
        # x contains the metadata vector and the bias vector -> 2 * metadata_dim
        lstm_hidden_states = list()
        lstm_context_states = list()
        for i in range(len(self.lstm_hidden_encoders)):
            lstm_hidden_states.append(self.lstm_hidden_encoders[i](x))
            lstm_context_states.append(self.lstm_context_encoders[i](x))

        # (num_layers * num_directions = len(self.lstm_hidden_encoders)) * BS * out_state
        lstm_hidden_states = torch.stack(lstm_hidden_states)
        lstm_context_states = torch.stack(lstm_context_states)

        return lstm_hidden_states, lstm_context_states
