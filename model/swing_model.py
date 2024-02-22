import torch.nn as nn
import torch
from model.GRN import GRN
from model.MetadataEncoder import MetadataEncoder
from model.BiLSTM import TemporalEnricherLSTM
from model.TemporalFusion import TemporalFusionTransformer
from model.custom_loss.swing_loss import HourlySwingCertitudeAwareLoss

import math


class HourlySwingModel(nn.Module):
    def __init__(self, inp_dim, metadata_dim, metadata_bias, metadata_gate_bias, temporal_enricher_dropout,
                 fusion_model_dim, fusion_num_heads, fusion_num_layers=2, fusion_apply_grn=True, fusion_dropout=0.2,
                 max_window=336, positional_info=("grn",),
                 lstm_num_layers=1, lstm_bidirectional=True, lstm_dropout=0.2,
                 loss_punish_cert=1.2, device="cpu"):
        super(HourlySwingModel, self).__init__()

        self.device = device
        self.value_only = True
        self.positional_info = positional_info
        # +1 for metadata token
        self.max_window = max_window + 1

        # decide who will process raw input candles
        if "lstm" in self.positional_info:
            if lstm_bidirectional:
                assert fusion_model_dim % 2 == 0
                self.hid_dim = int(fusion_model_dim / 2)
            else:
                self.hid_dim = fusion_model_dim

            assert self.hid_dim % fusion_num_heads == 0

            self.metadata_encoder = MetadataEncoder(metadata_dim=metadata_dim,
                                                    inter_dim=4 * metadata_dim,
                                                    out_state=self.hid_dim,
                                                    num_state_layers=lstm_num_layers,
                                                    state_bidirectional=lstm_bidirectional,
                                                    out_trans=fusion_model_dim,
                                                    bias=metadata_bias,
                                                    gate_bias=metadata_gate_bias)

            self.temporal_enricher = TemporalEnricherLSTM(inp_dim=inp_dim,
                                                          hid_dim=self.hid_dim,
                                                          num_layers=lstm_num_layers,
                                                          bidirectional=lstm_bidirectional,
                                                          dropout=lstm_dropout)
        elif "grn" in self.positional_info:
            assert "sinusodial" in self.positional_info or "learnable" in self.positional_info
            # will encode the metadata and bias vector for transformer
            self.metadata_encoder = GRN(inp_dim_x=metadata_dim,  # for metadata
                                        inp_dim_m=metadata_dim,  # for bias vector
                                        inter_dim=4 * metadata_dim,
                                        out_dim=fusion_model_dim,
                                        bias=True,
                                        gate_bias=True,
                                        dual=True)

            # will transform the input candles with info from metadata for transformer
            # in fact it is not a temporal enricher but a metadata enricher
            self.temporal_enricher = GRN(inp_dim_x=inp_dim,
                                         inp_dim_m=metadata_dim,
                                         inter_dim=4 * metadata_dim,
                                         out_dim=fusion_model_dim,
                                         bias=True,
                                         gate_bias=True,
                                         dual=True)
        else:
            assert False, "'lstm' or 'grn' must be in positional_info"

        self.dropout = nn.Dropout(p=temporal_enricher_dropout)

        if "sinusodial" in self.positional_info:
            self.positional_embeddings = self._create_sinusoidal_embeddings(fusion_model_dim, self.max_window)
        elif "learnable" in self.positional_info:
            self.positional_embeddings = nn.Parameter(torch.zeros(self.max_window, fusion_model_dim))
        else:
            self.positional_embeddings = torch.zeros(self.max_window, fusion_model_dim).to(self.device)

        self.temporal_fusion = TemporalFusionTransformer(model_dim=fusion_model_dim,
                                                         ff_dim=4 * fusion_model_dim,
                                                         num_heads=fusion_num_heads,
                                                         num_layers=fusion_num_layers,
                                                         apply_grn=fusion_apply_grn,
                                                         dropout=fusion_dropout)
        self.value_head = nn.Linear(in_features=fusion_model_dim, out_features=1)
        self.certitude_head = nn.Sequential(
            nn.Linear(in_features=fusion_model_dim, out_features=1),
            nn.ReLU()
        )
        self.directional_bias_vectors = nn.Embedding(num_embeddings=2, embedding_dim=metadata_dim)

        self.criterion1 = nn.MSELoss()
        self.criterion2 = HourlySwingCertitudeAwareLoss(punish_cert=loss_punish_cert)

    def forward(self, inp):
        """
        x           -> BS * seq_len * OHLCV
        metadata    -> BS * metadata_dim
        label       -> BS * 2
        weekday     -> BS
        """
        # unpacks to: x, metadata, label, weekday, base_info (index point, scale factor, time_last_candle) ?=
        #                                                    (pw_mid, pw_half_range, time_last_candle)
        x, metadata, label, _, _ = inp
        """
        if self.train then go like this;
        else duplicate the data and give one of them positive bias, the other negative bias and so on
        """
        bias_vector = self.directional_bias_vectors(torch.div(label[:, 0].long() + 1, 2, rounding_mode='trunc'))
        if "lstm" in self.positional_info:
            init_hid_states, init_cell_states, token = self.metadata_encoder((metadata, bias_vector))
            # states    -> BS * hid_dim
            # token     -> BS * fusion_model_dim
            x = self.temporal_enricher(x, init_hid_states, init_cell_states)  # BS * seq_len * fusion_model_dim

        elif "grn" in self.positional_info:
            token = self.metadata_encoder((metadata, bias_vector))
            # in fact it is not a temporal enricher but a metadata enricher
            # make the dimension compatible with input candles by copying it
            seq_len = x.shape[1]
            metadata = metadata.unsqueeze(1).expand(-1, seq_len, -1)
            x = self.temporal_enricher((x, metadata))
        else:
            assert False

        token = token.unsqueeze(1)
        x_fusion = torch.cat((token, x), dim=1)

        # add positional embeddings
        x_fusion += self.positional_embeddings
        x_fusion = self.dropout(x_fusion)

        # BS * fusion_model_dim
        out_reg, out_std = self.temporal_fusion(x_fusion)

        if self.value_only:
            # BS * 1
            out_reg = self.value_head(out_reg).squeeze(1)
            assert out_reg.shape == label[:, 1].shape
            loss = self.criterion1(out_reg, label[:, 1])
            return out_reg, loss

        else:
            # BS * 1
            out_reg = self.value_head(out_reg).squeeze(1)
            out_std = self.certitude_head(out_std).squeeze(1)
            assert out_reg.shape == label[:, 1].shape
            loss = self.criterion2(out_reg, out_std, label)
            return out_reg, out_std, loss

    def set_value_only(self, value_only=True):
        self.value_only = value_only

    def _create_sinusoidal_embeddings(self, emb_size, max_len):
        pos_emb = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)
        pos_emb = pos_emb.to(self.device)
        return pos_emb
