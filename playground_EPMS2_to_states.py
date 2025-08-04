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
        min_dists += 1e-6 
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

        total = self.alpha * recon_loss + self.beta * mask_loss + 1e-6  
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
    

class UNet1DVariableDecoder(nn.Module):
    def __init__(self, in_channels, encoder_depth=4, decoder_depth=6, base_channels=64,
                 init_thresh=0.1, init_alpha=10.0, learn_alpha=False,
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

    def forward(self, x):
        signal = x[:, -1, :].unsqueeze(1).detach()  # shape: (B, 1, T)
        others = x[:, :-1, :]  # shape: (B, C-1, T)
        x = torch.cat((others, signal), dim=1)  # shape: (B, C, T)
        distance = x[:, 3, :].unsqueeze(1)  # shape: (B, 1, T)
        if self.use_DistanceGate_mask:
            distance_signal = self.gate(distance)  # shape: (B, 1, T)
            x *= distance_signal  # Apply gate to input

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
        self.initial_conv = nn.Conv1d(128, base_channels, kernel_size=3, padding=1)
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


def clear_cuda_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def train_epoch(model_unet, model_decoder, dataloader, optimizer, criterion, device):
    model_unet.train()
    model_decoder.train()
    total_loss = 0
    total_recon_loss = 0
    total_mask_loss = 0

    for i, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()

        output_target = y_batch[:, 0, :]
        state_target = y_batch[:, 1, :]

        binary_output = model_unet(X_batch)
        output = model_decoder(binary_output)
        output = output.clamp(-20, 20)  # logits are raw, keep in safe range
        binary_output = torch.sigmoid(binary_output)
        loss, recon_loss, mask_loss = criterion(output, state_target, binary_output)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN/Inf detected in loss at batch {i}. Skipping update.")

            # print output and binary where nan or inf is detected
            if torch.isnan(recon_loss).any() or torch.isinf(recon_loss).any():
                print(f"⚠️ NaN/Inf detected in recon_loss at batch {i}.")
            if torch.isnan(mask_loss).any() or torch.isinf(mask_loss).any():
                print(f"⚠️ NaN/Inf detected in mask_loss at batch {i}.")
            clear_cuda_memory()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_unet.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model_decoder.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_mask_loss += mask_loss.item()

        del X_batch, y_batch, binary_output, output
        del loss, recon_loss, mask_loss
        clear_cuda_memory()

    return total_loss / len(dataloader), total_recon_loss / len(dataloader), total_mask_loss / len(dataloader)


@torch.no_grad()
def validate_epoch(model_unet, model_decoder, dataloader, criterion, 
                   best_val_loss=float('inf'), best_model_state=None,
                   device='cpu'):
    model_unet.eval()
    model_decoder.eval()

    total_loss = 0
    val_preds = []
    val_targets = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        output_target = y_batch[:, 0, :]
        state_target = y_batch[:, 1, :]

        binary_output = model_unet(X_batch)
        output = model_decoder(binary_output)
        output = output.clamp(-20, 20)  # logits are raw, keep in safe range

        binary_output = torch.sigmoid(binary_output)
        val_loss, _, _ = criterion(output, state_target, binary_output)

        total_loss += val_loss.item()
        val_preds.append(torch.sigmoid(output).squeeze(1).cpu())
        val_targets.append(output_target.cpu())

        clear_cuda_memory()
        del X_batch, y_batch, binary_output, output, val_loss

    total_val_loss = total_loss / len(dataloader)
    if total_val_loss < best_val_loss:
        print(f"New best validation loss: {total_val_loss:.4f} at best_val_loss {best_val_loss}")
        best_val_loss = total_val_loss
        best_model_state = {
            'model_unet': model_unet.state_dict(),
            'model_decoder': model_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': total_val_loss
        }
        torch.save(best_model_state, "best_model_EPMS2_to_states.pt")
    
    
    last_model_state = {
            'model_unet': model_unet.state_dict(),
            'model_decoder': model_decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': total_val_loss
        }
    torch.save(last_model_state, "last_model_EPMS2_to_states.pt")

    val_preds = torch.cat(val_preds).numpy().flatten()
    val_targets = torch.cat(val_targets).numpy().flatten()
    return total_val_loss, val_preds, val_targets, best_val_loss



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
fulltime_padded_states = np.clip(fulltime_padded_states, 0, 1)

print(f"Number of samples: {len(y)}")
print('fulltime_padded_states', len(fulltime_padded_states_all), fulltime_padded_states_all[0].shape)
print(f"Shape of C: {C[0].shape}, Shape of D: {len(D[0])}, Shape of y: {y[0].shape}")

plt.figure(figsize=(10, 4))
plt.plot(y[0])
plt.title("Distance between EPs")
plt.xlabel("Time")
plt.ylabel("Distance")
plt.show()

y = torch.stack(y).to(device)

X = torch.stack(C).to(device)  # [N, 3, T_in] 
D = torch.tensor(D, dtype=torch.float32).to(device)  # [N, T_in]
X = torch.cat((X, D.unsqueeze(1)), dim=1)  # [N, 4, T_in]
X = torch.cat((X, y.unsqueeze(1)), dim=1)  # [N, 4, T_in]



print(X.shape, y.shape)

N_in = X.shape[1]   # Number of input channels +1 (for distance gate)
N_out = 1  # Number of output channels (MS2 signal)
T_in = y.shape[1]
model_unet = UNet1DVariableDecoder(N_in, encoder_depth=2, decoder_depth=2, base_channels=8,
                                   init_thresh=0.1, init_alpha=1.0, learn_alpha=False,
                                   use_DistanceGate_mask=False)
model_decoder = CNNReconstructorResidual(out_channels=N_out, output_length=T_in)

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
pos_weight_train = np.clip(pos_weight_train, 0.01, 100)

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

criterion = JointLoss(
    recon_loss_fn=nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight_train]).to(device)),
    mask_loss_fn=EntropyLoss(),
    #mask_loss_fn=NormalizedCutLoss(k=2),
    alpha=1.0,
    beta=0.0
)

state_loss_criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight_train]).to(device))

optimizer = torch.optim.Adam(
    list(model_unet.parameters()) + list(model_decoder.parameters()),
      lr=1e-3)


# ----- Training Loop -----
num_epochs = 50
train_losses = []
val_losses = []

best_val_loss = float('inf')  # initialize to a large value
for epoch in trange(num_epochs, desc="Training Progress"):
    train_loss, recon_loss, mask_loss = train_epoch(model_unet, model_decoder, train_loader, optimizer, criterion, device)
    val_loss, val_preds, val_targets, best_val_loss = validate_epoch(model_unet, model_decoder, val_loader, criterion, 
                                                      best_val_loss=best_val_loss, best_model_state=None,
                                                      device=device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print("Learned threshold distance:", model_unet.gate.dc.item())
    print("Learned alpha distance:", model_unet.gate.alpha.item())

    print(f"[Epoch {epoch+1}/{num_epochs}]  🔧 Train Loss: {train_loss:.4f}, Recon Loss: {recon_loss:.4f}, Mask Loss: {mask_loss:.4f} | 🧪 Val Loss: {val_loss:.4f}")

print("Learned threshold distance:", model_unet.gate.dc.item())


plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# %%
torch.cuda.empty_cache()


print(X_val.shape, y_val.shape)
checkpoint = torch.load("last_model_EPMS2_to_states.pt", map_location='cpu')
model_unet.load_state_dict(checkpoint['model_unet'])
model_decoder.load_state_dict(checkpoint['model_decoder'])

print("Learned threshold distance:", model_unet.gate.dc.item())
print("Learned alpha distance:", model_unet.gate.alpha.item())

print(X_val.shape, y_val.shape)
checkpoint = torch.load("best_model_EPMS2_to_states.pt", map_location='cpu')
model_unet.load_state_dict(checkpoint['model_unet'])
model_decoder.load_state_dict(checkpoint['model_decoder'])

print("Learned threshold distance:", model_unet.gate.dc.item())
print("Learned alpha distance:", model_unet.gate.alpha.item())


model_unet.eval()
model_decoder.eval()
model_unet.to('cpu')
model_decoder.to('cpu')

binary = model_unet(X_test.to('cpu'))  # [B, 1, T_out]
output = model_decoder(binary.to('cpu'))  # [B, N, T_in]
binary = torch.sigmoid(binary)  # Apply sigmoid to get probabilities
output = torch.sigmoid(output)  # Apply sigmoid to get probabilities
output_binary = output > 0.5  # Apply sigmoid to get probabilities
binary0 = binary.squeeze(1)  # [B, T_out]
output = output.squeeze(1)  # [B, T_in]



x = torch.arange(0,1,0.01)
y = torch.sigmoid(model_unet.gate.alpha.item() * (model_unet.gate.dc.item() - x))
plt.figure(figsize=(5, 4))
plt.plot(x, y, label='Sigmoid Gate Function')
plt.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.xlabel('Distance')
plt.ylabel('Gate Output')
plt.title('Distance Gate Function')
plt.legend()


print(f"Binary output shape: {binary0.shape}")
print(f"Output shape: {output.shape}")
print(f"X_test shape: {X_test.shape}")
# plot some outputs versus targets

idx_chosen = np.random.randint(0, len(X_val))


fig, axs = plt.subplots(4,1,figsize=(12, 9))

axs[0].plot(X_test[idx_chosen,3,:].cpu().numpy().flatten(), label='Signal', color='C0')
axs[0].plot(X_test[idx_chosen,3,:].cpu().numpy().flatten()<=0.1, label='Under radius', color='k')

#axs[0].plot(X_test[i,4,:].cpu().numpy().flatten(), label='GT State', color='C1')
axs[0].plot(output[idx_chosen].cpu().detach().numpy().flatten(), label='Pred. State', color='C2')
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Signal Value")
axs[0].set_title(f"Residual of Output vs Binary sequence for X_test {idx_chosen+1}")
axs[0].legend()

axs[1].plot(X_test[idx_chosen,4,:].cpu().numpy().flatten(), label='Signal', color='C0')
axs[1].plot(states_all_test[idx_chosen].cpu().numpy().flatten(), label='GT State', color='C1')
axs[1].plot(output_binary[idx_chosen].cpu().detach().numpy().flatten(), label='Pred. State', color='C2')
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Signal Value")
axs[1].set_title(f"Residual of Output vs Binary sequence for X_test {idx_chosen+1}")
axs[1].legend()

axs[2].plot(X_test[idx_chosen,3,:].cpu().numpy().flatten()<=0.1, label='Under radius', color='k')
axs[2].plot(output[idx_chosen].cpu().detach().numpy().flatten(), label='Pred. State', color='C2')
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Signal Value")
axs[2].set_title(f"Residual of Output vs Binary sequence for X_test {idx_chosen+1}")
axs[2].legend()

axs[3].plot(states_all_test[idx_chosen].cpu().numpy().flatten(), label='GT State', color='C1')
axs[3].plot(output[idx_chosen].cpu().detach().numpy().flatten(), label='Pred. State', color='C2')
axs[3].set_xlabel("Time")
axs[3].set_ylabel("Signal Value")
axs[3].set_title(f"Residual of Output vs Binary sequence for X_test {idx_chosen+1}")
axs[3].legend()

plt.tight_layout()
plt.show()


def find_segments(inarray):
    """ 
    input: predicted labels, diffusion labels shape = (n,)
    output: segment run lengths, start positions of segments, difftypes of segments
    """
    ia = np.asarray(inarray)                # force numpy
    n = len(ia)
    if n == 0: 
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]             # pairwise unequal (string safe)
        i = np.append(np.where(y), n-1)   # must include last element posi
        z = np.diff(np.append(-1, i))     # run lengths
        p = np.cumsum(np.append(0, z)) # positions
        return(z, p, ia[i])


states_all_test_seglengths = []
for i in range(len(states_all_test)):
    inarray = states_all_test[i].cpu().detach().numpy().flatten()
    segment_lengths, segment_starts, segment_types = find_segments(inarray)
    states_all_test_seglengths.append(segment_lengths)
states_all_test_seglengths_flat = np.concatenate(states_all_test_seglengths)


output_binary_seglengths = []
for i in range(len(output_binary)):
    inarray = output_binary[i].cpu().detach().numpy().flatten()
    segment_lengths, segment_starts, segment_types = find_segments(inarray)
    output_binary_seglengths.append(segment_lengths)
output_binary_seglengths_flat = np.concatenate(output_binary_seglengths)
    

plt.figure(figsize=(5, 4))
plt.hist(states_all_test_seglengths_flat, bins=100, alpha=0.5, label='GT State Segments')
plt.hist(output_binary_seglengths_flat, bins=100, alpha=0.5, label='Pred. Binary Segments')
plt.xlabel('Segment Length')
plt.ylabel('Frequency')
plt.title('Segment Length Distribution')
plt.legend(loc='upper right')

plt.annotate(f'Avg. seg. length GT: {np.mean(states_all_test_seglengths_flat):.2f}\n'
             f'Avg. seg. length Pred: {np.mean(output_binary_seglengths_flat):.2f}',
             xy=(0.73, 0.8), xycoords='axes fraction',
             fontsize=10, ha='left', va='top')
plt.show()


# Acc metrics for each sequence

Acc_all = []
for i in range(len(output_binary)):
    acc = (output_binary[i].cpu().detach().numpy().flatten() == states_all_test[i].cpu().numpy().flatten()).mean() 
    Acc_all.append(acc)


plt.figure(figsize=(5, 4))
plt.hist(Acc_all, bins=10, alpha=0.7, color='C0')

plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Accuracy Distribution for Each Sequence')
plt.axvline(np.mean(Acc_all), color='C1', linestyle='--', label=f'Mean Accuracy: {np.mean(Acc_all):.2f}')
plt.legend()
plt.xlim(0., 1)
plt.show()

# %%
