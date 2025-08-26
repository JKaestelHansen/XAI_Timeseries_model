import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from captum.attr import Saliency, IntegratedGradients
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm import trange
import pickle


# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input tensor.
    The simple sin/cos positional encoding is used.
    10000.0 is used as per "Attention is all you need"
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# -------------------------
# Transformer Model
# -------------------------
class TransformerModel(nn.Module):
    def __init__(self, 
                 vocab_size=1, 
                 d_model=6, 
                 nhead=2, 
                 num_encoder_layers=2,
                 num_decoder_layers=2, 
                 dim_feedforward=6, 
                 dropout=0.1, 
                 max_len=1000,
                 layer_norm_eps=1e-5,
                 activation=torch.nn.functional.relu,
                 batch_first=True,
                 norm_first=False,
                 bias=True,
                 device='cuda',
                 dtype=torch.float32
                 ):
        super().__init__()
        self.model_type = "Transformer"
        if dtype == torch.float32:
            self.embedding = nn.Linear(vocab_size, d_model)
        else:
            self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        factory_kwargs = {"device": device, "dtype": dtype}

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
                bias,
                **factory_kwargs,)
        encoder_norm = nn.LayerNorm(
                d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.encoder = nn.TransformerEncoder(
                encoder_layer, num_encoder_layers, encoder_norm)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                dropout,
                activation,
                layer_norm_eps,
                batch_first,
                norm_first,
                bias,
                **factory_kwargs,)
        decoder_norm = nn.LayerNorm(
                d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.decoder = nn.TransformerDecoder(
                decoder_layer, num_decoder_layers, decoder_norm)

        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            custom_encoder=self.encoder,
            custom_decoder=self.decoder,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # ensures input is [batch, seq, feature]
        )
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None,
                src_padding_mask=None, 
                tgt_padding_mask=None, 
                src_is_causal=None,
                tgt_is_causal=None,
                memory_key_padding_mask=None):

        src_emb = self.pos_encoder(self.embedding(src))

        if tgt is None:
            output = self.encoder(src_emb,
                                mask=src_mask,
                                src_key_padding_mask=src_padding_mask)
        if tgt is not None:
            tgt_emb = self.pos_encoder(self.embedding(tgt))
            output = self.transformer(
                src_emb, tgt_emb,
                src_mask=src_mask, 
                tgt_mask=tgt_mask,
                src_is_causal=src_is_causal,
                tgt_is_causal=tgt_is_causal,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        return self.fc_out(output)



# Load Henrik data
def Load_simulated_data(path='_data/dataset_for_Jacob.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    C = []
    D = []
    y = []
    fulltime_padded_states_all = []
    for index in range(len(data)):
        (observation_times,
        ep_sep_obs_noisy,
        MS2_signal_noisy,
        state_times,
        state_sequence,
        ep_sep_obs,
        thinned_on_times,
        on_times,
        pol2_loading_events) = data[index]

        dist = np.linalg.norm(ep_sep_obs_noisy, axis=0)
        dist = dist / dist.max()  # Normalize distance
        D.append(dist)

        C.append(torch.tensor(ep_sep_obs_noisy, dtype=torch.float32))

        MS2_signal_noisy = MS2_signal_noisy / MS2_signal_noisy.max()  # Normalize
        MS2_signal_noisy = MS2_signal_noisy.astype(np.float32)
        y.append(torch.tensor(MS2_signal_noisy, dtype=torch.float32))

        padded_times = np.append(state_times, np.max(observation_times))
        padded_states = np.append(state_sequence, state_sequence[-1])

        fulltime_padded_states = np.zeros_like(observation_times)
        converted_padded_times = 30 * np.round(padded_times/30)
        for i in range(len(converted_padded_times)-1):
            start_time = converted_padded_times[i]
            end_time = converted_padded_times[i + 1]
            mask = (observation_times >= start_time) & (observation_times < end_time)
            fulltime_padded_states[mask] = padded_states[i]
        fulltime_padded_states_all.append(fulltime_padded_states)
    fulltime_padded_states = np.clip(fulltime_padded_states, 0, 1)

    return C, D, y, fulltime_padded_states_all


# ----- Positional Encoding -----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


# ----- Encoder -----
# Custom encoder layer to expose attention weights
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # So we get per-head attention
        )
        self.attn_weights = attn_weights.detach()  # Shape: [batch, num_heads, tgt_len, src_len]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# ----- Transformer Model -----
class Transformer(nn.Module):
    def __init__(self, seq_len, d_model=6, nhead=1, num_layers=1):
        super().__init__()
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, 2)

    def forward(self, x):
        x_embed = self.embedding(x)
        x_embed = self.pos_encoder(x_embed)
        if x_embed.requires_grad:
            x_embed.retain_grad()
        self.last_input = x_embed
        x_trans = x_embed
        self.attn_weights_all_layers = []
        for layer in self.layers:
            x_trans = layer(x_trans)
            self.attn_weights_all_layers.append(layer.attn_weights)
        return self.fc(x_trans)


# ----- Data stuff -----
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        """
        X: [N, T, features]
        y: [N, T, 1] or [N, T] (regression target)
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def generate_data(n_samples=1000, seq_len=100):
    X = torch.empty(n_samples, seq_len, 1)
    y = torch.empty(n_samples, seq_len, 1)

    for i in range(n_samples):
        N1 = torch.cumsum(torch.randn(seq_len, 1), dim=0)
        N2 = N1 + torch.normal(0, .4, (seq_len, 1))
        X[i] = ((N2-N1)**2)**0.5
        y[i] = ((X[i] < .05).float()) + torch.normal(0, .05, (seq_len, 1))
    return X, y
