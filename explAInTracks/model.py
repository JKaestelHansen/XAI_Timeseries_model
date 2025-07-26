import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_input_dims=2, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1, regression=False, n_classes=4):
        super().__init__()
        self.embedding = nn.Linear(n_input_dims, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers
        )
        self.regression = regression
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1 if regression else n_classes)
        )
        self.attn_weights = None
        self._register_hooks()

    def _register_hooks(self):
        def save_attention(module, input, output):
            self.attn_weights = module.self_attn.attn_output_weights.detach().cpu()
        self.transformer.layers[-1].register_forward_hook(save_attention)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        if mask is not None:
            lengths = (~mask).sum(1).unsqueeze(-1)
            x = (x * (~mask.unsqueeze(-1))).sum(dim=1) / lengths
        else:
            x = x.mean(dim=1)
        return self.output_head(x)

    def get_attention_scores(self):
        return self.attn_weights
