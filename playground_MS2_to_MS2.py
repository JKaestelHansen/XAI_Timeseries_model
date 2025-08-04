# %%
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharpnessLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred):
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        p = pred.clamp(self.epsilon, 1. - self.epsilon)
        entropy = - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return -entropy.mean()  # maximize sharpness


class EntropyLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred):
        """
        pred: Tensor of shape (B, 1, T) or (B, T) with values in [0, 1]
              typically from a sigmoid activation
        Returns average binary entropy across all elements
        """
        if pred.dim() == 3:
            pred = pred.squeeze(1)  # shape: (B, T)

        p = pred.clamp(self.epsilon, 1. - self.epsilon)
        entropy = - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return entropy.mean()
    

class NormalizedCutLoss(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.k = k

    def forward(self, pred):
        """
        pred: (B, 1, T) or (B, T)
        Encourages values to cluster around `k` centers (e.g. 0 and 1).
        """
        if pred.dim() == 3:
            pred = pred.squeeze(1)  # (B, T)
        means = torch.linspace(0, 1, steps=self.k).to(pred.device)  # [K]
        pred_flat = pred.view(pred.size(0), -1)  # (B, T)
        # Distance of each value to each center
        dists = (pred_flat.unsqueeze(-1) - means) ** 2  # (B, T, K)
        min_dists, _ = dists.min(dim=-1)  # (B, T)
        return min_dists.mean()


class JointLoss(nn.Module):
    def __init__(self, 
                 recon_loss_fn=nn.MSELoss(),
                 mask_loss_fn=NormalizedCutLoss(k=2),
                 alpha=1.0,
                 beta=0.1):
        super().__init__()
        self.recon_loss_fn = recon_loss_fn
        self.mask_loss_fn = mask_loss_fn
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target, binary_output):
        """
        output: model_decoder(binary_output) => shape (B, 1, T)
        target: ground truth => shape (B, T) or (B, 1, T)
        binary_output: output from model_unet => shape (B, 1, T)
        """
        if target.dim() == 2:
            target = target.unsqueeze(1)

        recon_loss = self.recon_loss_fn(output, target)
        mask_loss = self.mask_loss_fn(binary_output)

        total = self.alpha * recon_loss + self.beta * mask_loss
        return total, recon_loss, mask_loss
    

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
    def __init__(self, in_channels, encoder_depth=4, decoder_depth=6, base_channels=64):
        super().__init__()
        assert decoder_depth >= encoder_depth, "Decoder must be at least as deep as encoder"

        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth

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

        out = self.final_conv(x)
        out = torch.sigmoid(out)
        return out
    

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

class CNNReconstructorResidual(nn.Module):
    def __init__(self, out_channels, output_length, base_channels=64):
        super().__init__()
        self.output_length = output_length
        self.initial_conv = nn.Conv1d(1, base_channels, kernel_size=3, padding=1)
        self.res_block1 = ResidualBlock1D(base_channels)
        self.res_block2 = ResidualBlock1D(base_channels)
        self.final_conv1 = nn.Conv1d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x shape: [B, 1, T_out]
        x = self.initial_conv(x)    # [B, C, T_out]
        x = self.res_block1(x)      # [B, C, T_out]
        x = self.res_block2(x)      # [B, C, T_out]

        # Downsample/interpolate to T_in at the very end
        x = F.interpolate(x, size=self.output_length, mode='linear', align_corners=True)  # [B, out_channels, T_in]
        x = self.final_conv1(x)      # [B, out_channels, T_out]
        return x



torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('data/dataset_for_Jacob.pkl', 'rb') as f:
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

print(f"Number of samples: {len(y)}")
print('fulltime_padded_states', len(fulltime_padded_states_all), fulltime_padded_states_all[0].shape)
print(f"Shape of C: {C[0].shape}, Shape of D: {len(D[0])}, Shape of y: {y[0].shape}")

plt.figure(figsize=(10, 4))
plt.plot(y[0])
plt.title("Distance between EPs")
plt.xlabel("Time")
plt.ylabel("Distance")
plt.show()

X = torch.stack(y).unsqueeze(1).to(device)  # [N, 1, T_in] 
y = torch.stack(y)

print(X.shape, y.shape)

B, N, T_in = 8, 1, y.shape[1]
model_unet = UNet1DVariableDecoder(N, encoder_depth=2, decoder_depth=2, base_channels=2)
model_decoder = CNNReconstructorResidual(out_channels=N, output_length=T_in)

model_unet.to(device)
model_decoder.to(device)

# Train/val split
X_idx = np.arange(len(X))
X_train_idx, X_test_idx = train_test_split(X_idx, test_size=0.2, random_state=42)
X_train_idx, X_val_idx = train_test_split(X_train_idx, test_size=0.25, random_state=42)

X_train = X[X_train_idx].to(device)  # [N_train, 1, T_in]
X_val = X[X_val_idx].to(device)      # [N_val, 1, T_in]
X_test = X[X_test_idx].to(device)    # [N_test, 1, T_in]

y_train = y[X_train_idx].to(device)  # [N_train, T_in]
y_val = y[X_val_idx].to(device)      # [N_val, T_in]
y_test = y[X_test_idx].to(device)    # [N_test, T_in

states_all_train = [fulltime_padded_states_all[i] for i in X_train_idx]
states_all_val = [fulltime_padded_states_all[i] for i in X_val_idx]
states_all_test = [fulltime_padded_states_all[i] for i in X_test_idx]
states_all_train = torch.tensor(states_all_train, dtype=torch.float32).to(device)  # [N_train, T_out]
states_all_val = torch.tensor(states_all_val, dtype=torch.float32).to(device)      # [N_val, T_out]
states_all_test = torch.tensor(states_all_test, dtype=torch.float32).to(device)    # [N_test, T_out]

num_pos_train = torch.sum(states_all_train > 0.5).item()  # Count positive states
num_neg_train = torch.sum(states_all_train < 0.5).item()  # Count positive states
pos_weight_train = num_neg_train / num_pos_train

y_train = torch.stack([y_train, states_all_train], dim=1)  # [N_train, 2, T_in]
y_val = torch.stack([y_val, states_all_val], dim=1)      # [N_val, 2, T_in]
y_test = torch.stack([y_test, states_all_test], dim=1)    # [N_test, 2, T_in]

# Create Datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ----- Model, Loss, Optimizer -----
criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()

print('I could do a warm start where the model can go nuts of reconstruction and model isnt penalized for mask but then slowly amp up mask loss')
criterion = JointLoss(
    recon_loss_fn=nn.MSELoss(),
    mask_loss_fn=EntropyLoss(),  # or EntropyLoss(), SharpnessLoss() or NormalizedCutLoss(k=2)
    #mask_loss_fn=NormalizedCutLoss(k=2),
    alpha=1.0,
    beta=0.05
)

state_loss_criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight_train]).to(device))

optimizer = torch.optim.Adam(
    list(model_unet.parameters()) + list(model_decoder.parameters()),
      lr=1e-3)

val_losses = []
train_losses = []
for epoch in trange(10, desc="Training"):
    model_unet.train()
    model_decoder.train()

    epoch_loss = 0.0
    for i, (X_batch, y_batch) in enumerate(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()

        output_target = y_batch[:, 0, :]
        state_target = y_batch[:, 1, :]

        # Forward pass through UNet (outputs binary mask)
        binary_output = model_unet(X_batch)  # [B, 1, T_out]
        
        if i==10:
            print(np.unique(np.round(torch.flatten(binary_output).detach().cpu().numpy()), return_counts=True))

        # Forward pass through CNN reconstructor
        output = model_decoder(binary_output)  # [B, N, T_in]

        # Compute reconstruction loss only on recon vs input
        loss, recon_loss, cut_loss = criterion(output, output_target, binary_output)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        del binary_output, output, output_target  # Clear memory
        del X_batch, y_batch  # Clear memory
        del loss, recon_loss, cut_loss  # Clear memory
        torch.cuda.empty_cache()  # Clear GPU memory
        torch.cuda.ipc_collect()  # Collect IPC memory

    train_losses.append(epoch_loss / len(train_loader))

    # ----- Validation -----
    model_unet.eval()
    model_decoder.eval()

    val_preds = []
    val_targets = []
    val_loss_total = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            output_target = y_batch[:, 0, :]
            state_target = y_batch[:, 1, :]

            # Forward pass through UNet (outputs binary mask)
            binary_output = model_unet(X_batch)  # [B, 1, T_out]

            # Forward pass through CNN reconstructor
            val_output = model_decoder(binary_output)  # [B, N, T_in]

            # Compute reconstruction loss only on recon vs input
            loss, recon_loss, cut_loss = criterion(val_output, output_target, binary_output)
            
            val_loss_total += loss
            val_preds.append(torch.sigmoid(val_output).squeeze(-1).cpu())
            val_targets.append(output_target.cpu())

            del binary_output, val_output  # Clear memory
            del X_batch, y_batch, output_target  # Clear memory
            del loss, recon_loss, cut_loss  # Clear memory
            torch.cuda.empty_cache()  # Clear GPU memory
            torch.cuda.ipc_collect()  # Collect IPC memory

    val_preds = torch.cat(val_preds).numpy().flatten()
    val_targets = torch.cat(val_targets).numpy().flatten()
    val_loss = val_loss_total / len(val_loader)

    val_losses.append(val_loss)


    print(f"Epoch {epoch+1}/{200}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}")

# clear gpu memory
torch.cuda.empty_cache()
torch.cuda.ipc_collect()


# %%


model_unet.eval()
model_decoder.eval()
model_unet.to('cpu')
model_decoder.to('cpu')

binary = model_unet(X_test.to('cpu'))  # [B, 1, T_out]
output = model_decoder(binary.to('cpu'))  # [B, N, T_in]
binary0 = binary.squeeze(1)  # [B, T_out]
output = output.squeeze(1)  # [B, T_in]


print(f"Binary output shape: {binary0.shape}")
print(f"Output shape: {output.shape}")
# plot some outputs versus targets

i = np.random.randint(0, len(X_val))


fig, axs = plt.subplots(3,1,figsize=(10, 9))

axs[0].plot(X_test[i].cpu().numpy().flatten(), label='Input', color='C0')
axs[0].plot(output[i].cpu().detach().numpy(), label='Output', color='C1')
axs[0].plot(binary0[i].cpu().detach().numpy().flatten(), label='Binary', color='C2')
axs[0].set_title(f"Predicted + Binary sequence vs Input X_test {i+1}")
axs[0].set_ylabel("Signal Value")
axs[0].set_xlabel("Time")
axs[0].legend()


axs[1].plot(binary0[i].cpu().detach().numpy().flatten(), label='flip. Binary', color='C2')
axs[1].plot(states_all_test[i].cpu().numpy().flatten(), label='State', color='C0')
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Signal Value")
axs[1].set_title(f"Residual of Output vs Binary sequence for X_test {i+1}")
axs[1].legend()


axs[2].plot(binary0[i].cpu().detach().numpy().flatten(), label='flip. Binary', color='C2')
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Signal Value")
axs[2].set_title(f"Residual of X_test vs Binary sequence for X_test {i+1}")
axs[2].legend()

plt.tight_layout()
plt.show()
# %%
