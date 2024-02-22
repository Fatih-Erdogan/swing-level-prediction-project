import torch.nn as nn
from model.GRN import GRN
from model.Gate import Gate


class TemporalFusionTransformer(nn.Module):
    def __init__(self, model_dim, ff_dim, num_heads, num_layers, apply_grn=True, dropout=0.2):
        super(TemporalFusionTransformer, self).__init__()
        self.model_dim = model_dim
        self.apply_grn = apply_grn
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads,
                                                    dim_feedforward=ff_dim, batch_first=True,
                                                    dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)
        if self.apply_grn:
            self.grn_reg = GRN(model_dim, 0, model_dim, model_dim, bias=True, gate_bias=True, dual=False)
            self.grn_std = GRN(model_dim, 0, model_dim, model_dim, bias=True, gate_bias=True, dual=False)
        else:
            self.gate_reg = Gate(model_dim, bias=True)
            self.gate_std = Gate(model_dim, bias=True)
            self.layer_norm_reg = nn.LayerNorm(model_dim, eps=1e-6)
            self.layer_norm_std = nn.LayerNorm(model_dim, eps=1e-6)

    def forward(self, x):
        # take the output for first sequence element of each batch
        batch_size = x.shape[0]
        x = self.encoder(x)
        # BS * seq_len * model_dim -> BS * 1 * model_dim -> BS * model_dim
        x = x[:, 0, :].view(batch_size, self.model_dim)
        if self.apply_grn:
            out_reg = self.grn_reg(x)
            out_std = self.grn_std(x)
        else:
            out_reg = self.gate_reg(x)
            out_reg = self.layer_norm_reg(out_reg + x)
            out_std = self.gate_std(x)
            out_std = self.layer_norm_std(out_std + x)

        return out_reg, out_std
