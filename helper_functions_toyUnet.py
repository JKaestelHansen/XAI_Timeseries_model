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
import torch.nn.functional as F


class CalibratedGaussianNLL(nn.Module):
    def __init__(self, lambda_var=1e-4,   # strength on variance penalty
                 target_logvar=None,      # e.g. math.log(estimated_noise**2) or None
                 lambda_prior=0.0,        # strength toward target_logvar
                 clip_logvar=(-10.0, 10.0),
                 warmup_steps=0,          # no var penalty until this step
                 total_steps=100000):     # for smooth annealing if you want
        super().__init__()
        self.lambda_var = lambda_var
        self.target_logvar = target_logvar
        self.lambda_prior = lambda_prior
        self.clip_logvar = clip_logvar
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def forward(self, mean, logvar, y):
        # Core NLL
        logvar = torch.clamp(logvar, *self.clip_logvar)
        var = torch.exp(logvar)
        nll = 0.5 * (logvar + (y - mean)**2 / var)

        # Cosine anneal multiplier from 0 -> 1 after warmup
        self.step += 1
        t = (self.step.item() - self.warmup_steps) / max(1, (self.total_steps - self.warmup_steps))
        anneal = 0.0 if self.step.item() < self.warmup_steps else 0.5 * (1 - torch.cos(torch.tensor(min(max(t,0.0),1.0) * 3.1415926535)))

        # Penalize large variance (pushes toward certainty unless errors justify it)
        var_penalty = self.lambda_var * torch.exp(logvar)

        # Optional prior toward a target log-variance (helps avoid trivial inflation)
        if self.target_logvar is not None and self.lambda_prior > 0:
            prior_penalty = self.lambda_prior * (logvar - self.target_logvar)**2
        else:
            prior_penalty = 0.0

        loss = nll + anneal * (var_penalty + prior_penalty)
        return loss.mean()
    

class GaussianNLLLoss(nn.Module):
    """
    Gaussian Negative Log-Likelihood Loss.
    Predicts both mean and log-variance, and computes the NLL loss.
    Can be used like nn.MSELoss().
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps  # to avoid log(0) or division by zero

    def forward(self, y_pred_mean, y_pred_logvar, y_true):
        # Ensure logvar is stable
        logvar = y_pred_logvar.clamp(min=-10, max=10)
        precision = torch.exp(-logvar)
        loss = 0.5 * (logvar + precision * (y_true - y_pred_mean)**2)
        return loss.mean()
    

# -------------------
# Unet
# -------------------

class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return self.relu(out + residual)
    


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet1DVariableDecoder(nn.Module):
    def __init__(self, in_channels, encoder_depth=4, decoder_depth=6, base_channels=64,
                 init_thresh=0.1, init_alpha=10.0, learn_alpha=False, output_length=601,
                 use_DistanceGate_mask=True):
        super().__init__()
        assert decoder_depth >= encoder_depth, "Decoder must be at least as deep as encoder"

        self.use_DistanceGate_mask = use_DistanceGate_mask
        self.output_length = output_length

        # Optional DistanceGate
        if self.use_DistanceGate_mask:
            self.gate = DistanceGate(init_thresh=init_thresh,
                                     init_alpha=init_alpha,
                                     learn_alpha=learn_alpha)

        # -------------------
        # Encoder
        # -------------------
        self.pool = nn.MaxPool1d(2)
        self.downs = nn.ModuleList()
        skip_channels = []
        ch = in_channels
        for _ in range(encoder_depth):
            out_ch = base_channels
            self.downs.append(ConvBlock1D(ch, out_ch))
            skip_channels.append(out_ch)
            ch = out_ch
            base_channels *= 2

        self.bottleneck = ConvBlock1D(ch, ch)

        # -------------------
        # Decoder
        # -------------------
        self.ups = nn.ModuleList()
        self.reduce_channels = nn.ModuleList()
        for i in range(decoder_depth):
            self.ups.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=True))
            if i < encoder_depth:
                concat_ch = ch + skip_channels[-1 - i]
            else:
                concat_ch = ch
            self.reduce_channels.append(nn.Conv1d(concat_ch, ch // 2, kernel_size=1))
            ch = ch // 2

        # -------------------
        # Final output
        # -------------------
        self.final_conv = nn.Conv1d(ch, 2, kernel_size=1)  # mean + logvar

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, C, T]

        # Split input (signal + distance)
        signal = x[:, -1:, :].detach()         # [B, 1, T]
        others = x[:, :-1, :]                  # [B, C-1, T]
        distance = x[:, 3:4, :]                # [B, 1, T]
        x = torch.cat([others, signal], dim=1)

        # Apply DistanceGate if enabled
        if self.use_DistanceGate_mask:
            distance_signal = self.gate(distance)
            x = x * distance_signal
        else:
            distance_signal = torch.zeros_like(distance)

        # -------------------
        # Encoder
        # -------------------
        encs = []
        for down in self.downs:
            x = down(x)
            encs.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # -------------------
        # Decoder
        # -------------------
        for i, up in enumerate(self.ups):
            x = up(x)
            if i < len(encs):
                skip = F.interpolate(encs[-1 - i], size=x.shape[-1], mode='linear', align_corners=True)
                x = torch.cat([x, skip], dim=1)
            x = self.reduce_channels[i](x)

        # Final output
        x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=True)
        out = self.final_conv(x)
        mean = out[:, 0:1, :]
        logvar = torch.clamp(torch.log(F.softplus(out[:, 1:2, :]) + 1e-6), min=-10, max=10)

        mean = mean.squeeze(1)
        logvar = logvar.squeeze(1)
        distance_signal = distance_signal.squeeze(1)
        return mean, logvar, distance_signal
    



class DistanceGate(nn.Module):
    def __init__(self, init_thresh=0.1, init_alpha=100.0, learn_alpha=True):
        super().__init__()
        self.dc = nn.Parameter(torch.tensor(init_thresh))
        if learn_alpha:
            self.alpha = nn.Parameter(torch.tensor(init_alpha))
        else:
            self.alpha = torch.tensor(init_alpha)

    def forward(self, distance):
        # distance: shape (B, 1, T)
        alpha = torch.clamp(self.alpha, 1.0, 100.0)
        dc = torch.clamp(self.dc, 0.005, 2.0)
        return torch.sigmoid(alpha * (dc - distance))
    

class UNet1DVariableDecoder_resnet(nn.Module):
    def __init__(self, in_channels, encoder_depth=4, decoder_depth=6, base_channels=64,
                 init_thresh=0.1, init_alpha=10.0, learn_alpha=False, output_length=601,
                 use_DistanceGate_mask=True):
        super().__init__()
        assert decoder_depth >= encoder_depth, "Decoder must be at least as deep as encoder"

        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth

        self.use_DistanceGate_mask=use_DistanceGate_mask
        self.gate = DistanceGate(init_thresh=init_thresh,
                                 init_alpha=init_alpha,
                                 learn_alpha=learn_alpha)

        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2)
        ch = in_channels
        skip_channels = []

        # Encoder
        for _ in range(encoder_depth):
            out_ch = base_channels
            self.downs.append(ConvBlock1D(ch, out_ch))
            skip_channels.append(out_ch)
            ch = out_ch
            base_channels *= 2

        self.bottleneck = ConvBlock1D(ch, ch)

        self.ups = nn.ModuleList()
        self.reduce_channels = nn.ModuleList()

        # Decoder
        for i in range(decoder_depth):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
                ConvBlock1D(ch, ch // 2)
            ))

            if i < encoder_depth:
                concat_ch = ch // 2 + skip_channels[-1 - i]
            else:
                concat_ch = ch // 2

            self.reduce_channels.append(nn.Conv1d(concat_ch, ch // 2, kernel_size=1))
            ch = ch // 2

        self.final_conv = nn.Conv1d(ch, 128, kernel_size=1)


        self.output_length = output_length
        self.initial_conv = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.res_block1 = ResidualBlock1D(64)
        self.res_block2 = ResidualBlock1D(64)
        self.final_conv1 = nn.Conv1d(64, 2, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, C, T]
        signal = x[:, -1, :].unsqueeze(1).detach()  # shape: (B, 1, T)
        others = x[:, :-1, :]  # shape: (B, C-1, T)
        distance = x[:, 3, :].unsqueeze(1)  # shape: (B, 1, T)
        x = torch.cat((others, signal), dim=1)  # shape: (B, C, T)
        if self.use_DistanceGate_mask:
            distance_signal = self.gate(distance)  # shape: (B, 1, T)
            x *= distance_signal   # Apply gate to input
        else:
            distance_signal = torch.zeros_like(distance)

        encs = []
        for down in self.downs:
            x = down(x)
            encs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for i, up in enumerate(self.ups):
            x = up(x)
            if i < len(encs):
                skip = F.interpolate(encs[-1 - i], size=x.shape[-1], mode='linear', align_corners=True)
                x = torch.cat([x, skip], dim=1)
            x = self.reduce_channels[i](x)

        # output
        x = self.final_conv(x)
        x = self.initial_conv(x)    # [B, C, T_out]
        x = self.res_block1(x)      # [B, C, T_out]
        x = self.res_block2(x)      # [B, C, T_out]

        # Downsample/interpolate to T_in at the very end
        x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=True)  # [B, out_channels, T_in]
        out = self.final_conv1(x)      # [B, out_channels, T_out]
        mean = out[:, 0:1, :]       # [B, 1, T_out]
        s = out[:, 1:2, :]     # [B, 1, T_out]
        var = F.softplus(s) + 1e-6
        logvar = torch.log(var)
        logvar = torch.clamp(logvar, min=-10, max=10)

        del x
        return mean, logvar, distance_signal  # Return both output and distance signal
    
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
            batch_first=batch_first  # ensures input is [batch, seq, feature]
        )
        self.fc_out = nn.Linear(d_model, 2)

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

        out = self.fc_out(output)
        mean = out[:, :, 0]
        logvar = torch.clamp(torch.log(F.softplus(out[:, :, 1]) + 1e-6), min=-10, max=10)

        return mean, logvar, np.nan  # placeholder for distance_signal



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
            CustomTransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=False)
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
    X = torch.empty(n_samples, seq_len, 5)
    y = torch.empty(n_samples, seq_len, 1)

    for i in range(n_samples):
        N1 = torch.cumsum(torch.randn(seq_len, 3), dim=0)
        N2 = N1 + torch.normal(0, .1, (seq_len, 3))
        D = (torch.sum((N2-N1)**2, axis=1))**0.5
        X[i][:,:3] = N1
        X[i][:,3] = D
        X[i][:,4] = 0
        y[i] = ((D.unsqueeze(-1) < .05).float()) + torch.normal(0, .05, (seq_len, 1))
    return X, y


