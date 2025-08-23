# %%
# UNet-VAE with Bernoulli latent space that expands time dimension
# Includes KL divergence and binary sampling for VAE behavior

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import trange
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Causal Conv ---
class CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=0, dilation=dilation)
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return super().forward(x)


# --- ConvBlock supporting causal or non-causal ---
class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, causal=False):
        super().__init__()
        conv = CausalConv1d if causal else nn.Conv1d
        self.block = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=3),
            nn.ReLU(),
            conv(out_channels, out_channels, kernel_size=3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)


# --- U-Net Encoder with Causal Option ---
class UNetBernoulliEncoder(nn.Module):
    def __init__(self, in_channels, encoder_depth=4, decoder_depth=6, base_channels=64, causal=False):
        super().__init__()
        assert decoder_depth >= encoder_depth

        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth

        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2)
        ch = in_channels
        skip_channels = []

        for _ in range(encoder_depth):
            out_ch = base_channels
            self.downs.append(ConvBlock1D(ch, out_ch, causal=causal))
            skip_channels.append(out_ch)
            ch = out_ch
            base_channels *= 2

        self.bottleneck = ConvBlock1D(ch, ch, causal=causal)

        self.ups = nn.ModuleList()
        self.reduce_channels = nn.ModuleList()

        for i in range(decoder_depth):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='linear', align_corners=True),
                ConvBlock1D(ch, ch // 2, causal=causal)
            ))

            concat_ch = ch // 2 + (skip_channels[-1 - i] if i < encoder_depth else 0)
            self.reduce_channels.append(nn.Conv1d(concat_ch, ch // 2, kernel_size=1))
            ch = ch // 2

        self.final_conv = nn.Conv1d(ch, 1, kernel_size=1)

    def forward(self, x):
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

        logits = self.final_conv(x)
        return logits

# --- CNN decoder from binary mask ---
class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class TransformerDecoderWrapper(nn.Module):
    def __init__(self, d_model, nhead, num_layers=2, dim_feedforward=256, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)

        # Initial positional encoding
        self.register_buffer("positional_encoding", self._generate_positional_encoding(max_len, d_model))

    def _generate_positional_encoding(self, length, d_model):
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, length, d_model)
        return pe

    def forward(self, tgt, memory):
        B, L, _ = tgt.shape
        if L > self.positional_encoding.size(1):
            # Regenerate PE dynamically
            self.positional_encoding = self._generate_positional_encoding(L, self.d_model).to(tgt.device)

        # Add positional encoding to target
        tgt = tgt + self.positional_encoding[:, :L, :]

        return self.transformer_decoder(tgt, memory)


class AdaptiveTransformerReconstructor(nn.Module):
    def __init__(self, out_channels, output_length, base_channels=64, 
                 embed_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.output_length = output_length
        self.initial_conv = nn.Conv1d(1, base_channels, kernel_size=3, padding=1)
        self.final_proj = nn.Linear(embed_dim, out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.res_blocks = nn.ModuleList()
        self.encoder_proj = nn.Conv1d(base_channels, embed_dim, kernel_size=1)

        self.transformer_decoder = TransformerDecoderWrapper(
            d_model=embed_dim,
            nhead=num_heads,
            num_layers=num_layers,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            max_len=output_length
        )

        self.decoder_input_embed = nn.Parameter(torch.randn(1, output_length, embed_dim))

    def forward(self, x):
        B, C, L_in = x.shape
        x = self.initial_conv(x)

        # Dynamically add residual blocks + pooling
        L = x.shape[-1]
        while L > self.output_length * 1.25:
            block = ResidualBlock1D(x.shape[1])
            self.res_blocks.append(block.to(x.device))
            x = block(x)
            x = self.pool(x)
            L = x.shape[-1]

        # Project encoder output to transformer input shape (B, S, E)
        memory = self.encoder_proj(x).permute(0, 2, 1)  # (B, L, E)

        # Generate decoder inputs (same shape as output length)
        tgt = self.decoder_input_embed.repeat(B, 1, 1)

        # Decode
        decoded = self.transformer_decoder(tgt, memory)  # (B, L_out, E)
        out = self.final_proj(decoded).permute(0, 2, 1)  # (B, out_channels, output_length)
        return out
    

class AdaptiveCNNReconstructor(nn.Module):
    def __init__(self, out_channels, output_length, base_channels=64):
        super().__init__()
        self.output_length = output_length
        self.initial_conv = nn.Conv1d(1, base_channels, kernel_size=3, padding=1)
        self.final_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Store the dynamically created residual blocks
        self.res_blocks = nn.ModuleList()
        L = input_length = output_length * 2  # Start with a length that allows for downsampling
        while L > self.output_length * 1.25:  # allow a small margin before interpolating
            block = ResidualBlock1D(x.shape[1])
            self.res_blocks.append(block.to(x.device))
            L = math.ceil(current_length / 2)

    def forward(self, x):
        B, C, L_in = x.shape
        x = self.initial_conv(x)

        # Dynamically add residual blocks + pooling until length is close to target
        for block in self.res_blocks:
            x = block(x)
            x = self.pool(x)

        # Final upsample + conv
        x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=True)
        x = self.final_conv(x)
        return x
    

class CNNReconstructorResidual(nn.Module):
    def __init__(self, out_channels, output_length, base_channels=64):
        super().__init__()
        self.output_length = output_length
        self.initial_conv = nn.Conv1d(1, base_channels, kernel_size=3, padding=1)
        self.res_block1 = ResidualBlock1D(base_channels)
        self.res_block2 = ResidualBlock1D(base_channels)
        self.final_conv = nn.Conv1d(base_channels, out_channels, kernel_size=1)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_block1(x)
        x = self.pool(x)
        x = self.res_block2(x)
        x = self.pool(x)
        x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=True)
        x = self.final_conv(x)
        return x
    

class SimpleAdaptiveCNN(nn.Module):
    def __init__(self, input_length, output_length, out_channels=1, base_channels=32):
        super().__init__()
        self.output_length = output_length
        self.input_length = input_length

        layers = []
        in_channels = 1
        current_length = input_length

        # Downsampling: Conv + MaxPool until sequence is close to or below output_length
        while current_length > output_length * 1.5:
            layers.append(nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2))
            current_length = math.ceil(current_length / 2)
            in_channels = base_channels

        self.encoder = nn.Sequential(*layers)
        self.final_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=True)
        x = self.final_conv(x)
        return x


class ResBlock1D(nn.Module):
    def __init__(self, channels, use_batchnorm=False):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm1d(channels) if use_batchnorm else nn.Identity()
        self.norm2 = nn.BatchNorm1d(channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + residual)


class ResNetStyleDecoder1D(nn.Module):
    def __init__(self,
                 input_length,
                 output_length,
                 out_channels=1,
                 base_channels=32,
                 capacity=1.0,
                 use_batchnorm=False,
                 num_down_blocks=None):
        super().__init__()

        self.input_length = input_length
        self.output_length = output_length
        self.out_channels = out_channels

        channels = int(base_channels * capacity)

        self.input_proj = nn.Conv1d(1, channels, kernel_size=3, padding=1)

        # Determine how many downsampling steps we need
        if num_down_blocks is None:
            print(input_length, output_length, input_length / output_length)
            num_down_blocks = max(1, math.ceil(math.log2(input_length / output_length))) if input_length > output_length else 0
        self.down_blocks = nn.ModuleList()

        for _ in range(num_down_blocks):
            block = nn.Sequential(
                ResBlock1D(channels, use_batchnorm=use_batchnorm),
                nn.MaxPool1d(kernel_size=2)
            )
            self.down_blocks.append(block)

        # Final resizing to exact output length
        self.final_resample = nn.Identity() if input_length == output_length else \
            nn.Upsample(size=output_length, mode='linear', align_corners=True)

        self.output_proj = nn.Conv1d(channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.input_proj(x)  # (B, 1, T) → (B, C, T)

        for block in self.down_blocks:
            x = block(x)

        x = self.final_resample(x)
        x = self.output_proj(x)
        return x


def gumbel_sigmoid_sample(logits, temperature=0.1):
    noise = torch.rand_like(logits)
    gumbel = -torch.log(-torch.log(noise + 1e-8) + 1e-8)
    y = torch.sigmoid((logits + gumbel) / temperature)
    return y


# --- Bernoulli sampling ---
def bernoulli_sample(logits, mode='gumbel_ste'):
    probs = torch.sigmoid(logits)

    if mode == 'st':  # straight-through estimator
        hard = (probs > 0.5).float()
        return hard + probs - probs.detach()
    
    elif mode == 'gumbel_ste':
        y_soft = gumbel_sigmoid_sample(logits, temperature=0.1)
        y_hard = (y_soft > 0.5).float()
        return y_hard + y_soft - y_soft.detach()  # STE

    elif mode == 'gumbel':  # differentiable approximation
        noise = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(noise + 1e-8) + 1e-8)
        return torch.sigmoid((logits + gumbel) / 0.1)

    elif mode == 'sample':  # true Bernoulli sampling (non-differentiable)
        return torch.bernoulli(probs)

    else:
        raise ValueError(f"Unknown mode: {mode}")

# --- KL divergence for Bernoulli latent ---
class BernoulliKLLoss(nn.Module):
    def __init__(self, prior=0.5, epsilon=1e-6):
        super().__init__()
        self.prior = prior
        self.eps = epsilon

    def forward(self, logits):
        p = torch.sigmoid(logits).clamp(self.eps, 1 - self.eps)
        kl = p * torch.log(p / self.prior) + (1 - p) * torch.log((1 - p) / (1 - self.prior))
        return kl.mean()


class BetaSchedule:
    def __init__(self, max_epochs=50, max_beta=0.1):
        self.max_epochs = max_epochs
        self.max_beta = max_beta

    def __call__(self, epoch):
        if epoch < int(self.max_epochs * 0.25):
            return 0.0
        elif epoch < int(self.max_epochs * 0.5):
            return self.max_beta * (epoch - int(self.max_epochs * 0.25)) / (self.max_epochs * 0.25)
        elif epoch < int(self.max_epochs * 0.75):
            return self.max_beta
        else:
            return self.max_beta

     
class JointVAELoss_schedule(nn.Module):
    def __init__(self, recon_loss_fn, kl_loss_fn, alpha=1.0, beta=1.0, 
                 sparsity_weight=0.1):
        super().__init__()
        self.recon_loss_fn = recon_loss_fn
        self.kl_loss_fn = kl_loss_fn
        self.alpha = alpha
        self.beta = beta
        self.sparsity_weight = sparsity_weight

    def forward(self, mean, target, logits, z, epoch=0):
        recon_loss = self.recon_loss_fn(mean, target)
        kl_loss = self.kl_loss_fn(logits)
        
        # Compute expected sparsity penalty on the latent probabilities
        sparsity_penalty = z.mean()

        beta_val = self.beta(epoch) if callable(self.beta) else self.beta

        total_loss = self.alpha * recon_loss + beta_val * kl_loss + self.sparsity_weight * sparsity_penalty

        return total_loss, self.alpha * recon_loss, beta_val * kl_loss, self.sparsity_weight * sparsity_penalty

# --- Load and preprocess data ---
with open('_data/dataset_for_Jacob.pkl', 'rb') as f:
    data = pickle.load(f)

y = []
full_states = []
full_pol2 = []
for entry in data:
    observation_times, _, signal, state_times, state_seq, _, _, _, pol2_loading_events = entry
    norm_signal = (signal / signal.max()).astype(np.float32)
    y.append(torch.tensor(norm_signal))

    
    padded_times = np.append(state_times, np.max(observation_times))
    padded_states = np.append(state_seq, state_seq[-1])

    padded = np.zeros_like(observation_times)
    converted_padded_times = 30 * np.round(padded_times/30)
    for i in range(len(converted_padded_times)-1):
        start_time = converted_padded_times[i]
        end_time = converted_padded_times[i + 1]
        mask = (observation_times >= start_time) & (observation_times < end_time)
        padded[mask] = padded_states[i]
    full_states.append(torch.tensor(padded, dtype=torch.float32))

    events = []
    for i, t in enumerate(observation_times):
        num_events = np.sum(pol2_loading_events <= t)
        events.append(num_events)
    full_pol2.append(torch.tensor(events, dtype=torch.float32))



# Stack and prepare data
X = torch.stack(y).unsqueeze(1).to(device)
y = torch.stack(y).to(device)
T_in = y.shape[1]

# Split data into training and validation sets
train_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.2)
X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

# Promoter states for later post-hoc interpretation
states = torch.stack(full_states).to(device)
states_train, states_val = states[train_idx], states[val_idx]

pol2_loadings = torch.stack(full_pol2).to(device)
pol2_loadings_train, pol2_loadings_val = pol2_loadings[train_idx], pol2_loadings[val_idx]

# Create DataLoader for training and validation
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

# --- Initialize model ---
encoder_depth = 2
decoder_depth = 3

num_z_samples = 20
epochs = 100


model_unet = UNetBernoulliEncoder(1, 
                                  encoder_depth=encoder_depth, 
                                  decoder_depth=decoder_depth, 
                                  base_channels=16,
                                  causal=False  # Use causal convolutions
                                  ).to(device)
#model_decoder = AdaptiveTransformerReconstructor(out_channels=1, output_length=T_in).to(device)

#model_decoder = AdaptiveCNNReconstructor(out_channels=1, output_length=T_in).to(device)

# model_decoder = SimpleAdaptiveCNN(input_length=int(T_in*2*(decoder_depth-encoder_depth)), output_length=T_in, out_channels=1).to(device)
upsampled = 2 * (decoder_depth - encoder_depth) if decoder_depth > encoder_depth else 1
model_decoder = ResNetStyleDecoder1D(
    input_length=T_in * upsampled,   # or latent length
    output_length=T_in,      # desired reconstruction size
    out_channels=1,
    base_channels=32,
    capacity=2.0,            # increase for more filters per layer
    use_batchnorm=True       # useful if latent is binary
).to(device)


criterion = JointVAELoss_schedule(
    recon_loss_fn=nn.MSELoss(),
    kl_loss_fn=BernoulliKLLoss(prior=0.5),
    alpha=5.0,
    beta=0.0,
    sparsity_weight=0.1
)
beta_sched = BetaSchedule(max_epochs=epochs, max_beta=0.01)


optimizer = torch.optim.Adam(
    list(model_unet.parameters()) + list(model_decoder.parameters()), 
    lr=1e-3)

# --- Training loop ---
train_losses = []
recon_losses = []
kl_losses = []
sparsity_losses = []

val_losses = []
val_recon_losses = []
val_kl_losses = []
val_sparsity_losses = []
best_val_loss = float('inf')
for epoch in trange(epochs):
    model_unet.train()
    model_decoder.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    total_sparsity_loss = 0

    current_beta = beta_sched(epoch)
    criterion.beta = current_beta

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        target = y_batch

        logits = model_unet(X_batch)
        probs = torch.sigmoid(logits)

        z = bernoulli_sample(logits)
        recon = model_decoder(z)

        loss, rloss, kloss, sparsity = criterion(recon, target.unsqueeze(1), logits, z, epoch=epoch)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_recon_loss += rloss.item()
        total_kl_loss += kloss.item()
        total_sparsity_loss += sparsity.item()

    train_losses.append(total_loss / len(train_loader))
    recon_losses.append(total_recon_loss / len(train_loader))
    kl_losses.append(total_kl_loss / len(train_loader))
    sparsity_losses.append(total_sparsity_loss / len(train_loader))
    

    # Validation
    model_unet.eval()
    model_decoder.eval()
    val_total_loss = 0
    val_recon_loss = 0
    val_kl_loss = 0
    val_sparsity_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            target = y_batch
            logits = model_unet(X_batch)

            z_tmp = torch.empty(num_z_samples, *logits.shape).to(device)
            for i in range(num_z_samples):
                z = bernoulli_sample(logits)
                z_tmp[i] = z
            z = z_tmp.mode(dim=0).values

            recon = model_decoder(z)
            loss, rloss, kloss, sparsity = criterion(recon, target.unsqueeze(1), logits, z, epoch=epoch)
            val_total_loss += loss.item()
            val_recon_loss += rloss.item()
            val_kl_loss += kloss.item()
            val_sparsity_loss += sparsity.item()

    epoch_val_loss = val_total_loss / len(val_loader)
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model_unet.state_dict(), 'best_model_unet.pth')
        torch.save(model_decoder.state_dict(), 'best_model_decoder.pth')

    val_losses.append(epoch_val_loss)
    val_recon_losses.append(val_recon_loss / len(val_loader))
    val_kl_losses.append(val_kl_loss / len(val_loader))
    val_sparsity_losses.append(val_sparsity_loss / len(val_loader))

    print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")
    print(f"\tTrain Recon Loss: {recon_losses[-1]:.4f}, Val Recon Loss: {val_recon_losses[-1]:.4f}")
    print(f"\tTrain KL Loss: {kl_losses[-1]:.4f}, Val KL Loss: {val_kl_losses[-1]:.4f}, beta: {current_beta:.4f}")
    print(f"\tTrain Sparsity Loss: {sparsity_losses[-1]:.4f}, Val Sparsity Loss: {val_sparsity_losses[-1]:.4f}")

# %%
fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
axs[0].plot(train_losses, label='Train Loss', color='C0')
axs[0].plot(val_losses, label='Val Loss', color='C1')
axs[0].set_title("Losses over Epochs")
axs[0].legend()
axs[1].plot(recon_losses, label='Train Recon Loss', color='C2')
axs[1].plot(val_recon_losses, label='Val Recon Loss', color='C3')
axs[1].set_title("Reconstruction Losses over Epochs")
axs[1].legend()
axs[2].plot(kl_losses, label='Train KL Loss', color='C4')
axs[2].plot(val_kl_losses, label='Val KL Loss', color='C5')
axs[2].set_title("KL Losses over Epochs")
axs[2].set_xlabel("Epoch")
axs[2].legend()
axs[3].plot(sparsity_losses, label='Train Sparsity Loss', color='C6')
axs[3].plot(val_sparsity_losses, label='Val Sparsity Loss', color='C7')
axs[3].set_title("Sparsity Losses over Epochs")
axs[3].set_xlabel("Epoch")
axs[3].legend()
plt.tight_layout()
plt.show()



# %%

idx_to_viz = np.random.randint(0, len(X_val), size=1)

# Load best model weights
model_unet.load_state_dict(torch.load('best_model_unet.pth'))
model_decoder.load_state_dict(torch.load('best_model_decoder.pth'))

model_unet.eval()
model_decoder.eval()

num_z_samples = 100  # Number of samples to average for Bernoulli sampling
with torch.no_grad():
    X_batch = X_val[idx_to_viz].to(device)
    target = y_val[idx_to_viz].to(device)

    logits = model_unet(X_batch)
    
    z_avg = torch.empty(num_z_samples, *logits.shape).to(device)
    for i in range(num_z_samples):
        z = bernoulli_sample(logits)
        z_avg[i] = z
    
    z_avg = z_avg.mode(dim=0).values

    recon = model_decoder(z_avg)

    recon = recon.cpu().numpy().squeeze()
    target = target.cpu().numpy().squeeze()
    probs = torch.sigmoid(logits).cpu().numpy().squeeze()
    logits = logits.cpu().numpy().squeeze()
    z = z_avg.cpu().numpy().squeeze()

print(logits.shape)
print(probs.shape)
print(recon.shape)
print(target.shape)
print(z_avg.shape)

upsample_ratio = 2*(decoder_depth-encoder_depth) if decoder_depth > encoder_depth else 1

z_times = np.arange(0, len(z)+1) / upsample_ratio  # Assuming 4 Hz sampling rate
z_events = [np.sum(z[:i+1]) for i in range(len(z))]  # Cumulative sum of Bernoulli events
z_events = np.append(0, z_events)

print(z)

print(np.arange(0, len(probs))/upsample_ratio)

fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
axs[0].plot(target, label='Target Signal', color='C1')
axs[0].plot(np.arange(0, len(probs))/upsample_ratio, probs, label='Sigmoid(Logits)', color='C5')
axs[0].set_title("Logits from UNet Encoder")

axs[1].plot(target, label='Target Signal', color='C1')
axs[1].plot(recon, label='Reconstructed Signal', color='C2')
axs[1].set_title("Reconstruction vs Target Signal")
axs[1].legend()

axs[2].plot(target, label='Target Signal', color='C1')
axs[2].plot(np.arange(0, len(probs))/upsample_ratio, z, label='Bernoulli Sample', color='C4')
axs[2].set_title("Bernoulli Sample from Logits")
axs[2].set_xlabel("Time")

axs[3].plot(pol2_loadings_val[idx_to_viz].cpu().numpy().squeeze(), 
            label='Pol II Loading Events', color='C3')
axs[3].set_title(f"Mode Bernoulli Samples (N={num_z_samples}) from Logits")
axs[3].set_xlabel("Time")
axs[3].legend()

axs[4].plot(z_times, z_events, label='Cumulative Bernoulli Events', color='C4')
axs[4].plot(pol2_loadings_val[idx_to_viz].cpu().numpy().squeeze(), 
            label='Pol II Loading Events', color='C3')
axs[4].set_title(f"Mode Bernoulli Samples (N={num_z_samples}) from Logits")
axs[4].set_xlabel("Time")
axs[4].legend()

axs[4].annotate('Difference in Pol II loading events {}'.format(
    np.sum(pol2_loadings_val[idx_to_viz].cpu().numpy().squeeze()[-1] - z_events[-1])), 
    xy=(0.81, 0.1), xycoords='axes fraction',
    fontsize=12, ha='center', va='center')

plt.tight_layout()
plt.show()


# %%
