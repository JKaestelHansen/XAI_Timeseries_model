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
    

def gaussian_nll_loss(y_pred_mean, y_pred_logvar, y_true):
    # y_pred_mean, y_pred_logvar: [B, 1, T]
    var = torch.exp(y_pred_logvar)
    return 0.5 * (y_pred_logvar + (y_true - y_pred_mean) ** 2 / var)


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
    


def clear_cuda_memory():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def train_epoch(model_unet, dataloader, optimizer, criterion, device):
    model_unet.train()
    
    total_loss = 0
    total_recon_loss = 0
    total_mask_loss = 0
    total_distance_signal_loss = 0
    for i, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()

        output_target = y_batch[:, 0, :]
        if output_target.dim() == 2:
            output_target = output_target.unsqueeze(1)

        mean, logvar, distance_signal = model_unet(X_batch)
        mean = mean.clamp(-20, 20)  # logits are raw, keep in safe range
        if model_unet.use_DistanceGate_mask:
            # do sparsity loss on distance_signal
            loss_distance_signal = torch.mean(distance_signal)
            loss_distance_signal *= 4
            total_distance_signal_loss += loss_distance_signal.item()

        sigmoid_mean = torch.sigmoid(mean)
        loss_entropy = 0.1 * EntropyLoss()(sigmoid_mean)

        #loss = criterion(mean, output_target) + 1e-6  
        loss = criterion(mean, logvar, output_target).mean() #+ loss_entropy
        loss += loss_distance_signal if model_unet.use_DistanceGate_mask else 0

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ NaN/Inf detected in loss at batch {i}. Skipping update.")
            # print output and binary where nan or inf is detected
            clear_cuda_memory()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_unet.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        del X_batch, y_batch, mean, logvar
        clear_cuda_memory()

    return total_loss / len(dataloader), total_recon_loss / len(dataloader), total_mask_loss / len(dataloader), total_distance_signal_loss / len(dataloader)


@torch.no_grad()
def validate_epoch(model_unet, dataloader, criterion, 
                   best_val_loss=float('inf'), best_model_state=None,
                   device='cpu'):
    model_unet.eval()

    total_loss = 0
    total_distance_signal_loss = 0
    val_preds = []
    val_targets = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        output_target = y_batch[:, 0, :]
        if output_target.dim() == 2:
            output_target = output_target.unsqueeze(1)

        mean, logvar, distance_signal = model_unet(X_batch)

        sigmoid_mean = torch.sigmoid(mean)
        loss_entropy = 0.1 * EntropyLoss()(sigmoid_mean)

        if model_unet.use_DistanceGate_mask:
            # do sparsity loss on distance_signal
            loss_distance_signal = torch.mean(distance_signal)
            loss_distance_signal *= 4
            total_distance_signal_loss += loss_distance_signal.item()

        # loss = criterion(mean, output_target)
        loss = criterion(mean, logvar, output_target).mean() #+ loss_entropy
        loss += loss_distance_signal if model_unet.use_DistanceGate_mask else 0

        total_loss += loss.item()
        val_preds.append(torch.sigmoid(mean).squeeze(1).cpu())
        val_targets.append(output_target.cpu())

        clear_cuda_memory()
        del X_batch, y_batch, mean, logvar

    total_val_loss = total_loss / len(dataloader)
    if total_val_loss < best_val_loss:
        best_val_loss = total_val_loss
        best_model_state = {
            'model_unet': model_unet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': total_val_loss
        }
        torch.save(best_model_state, "best_model_EPMS2_to_states.pt")
    
    
    last_model_state = {
            'model_unet': model_unet.state_dict(),
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

with open('_data/dataset_for_Jacob.pkl', 'rb') as f:
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

y = torch.stack(y).to(device)

X = torch.stack(C).to(device)  # [N, 3, T_in] 
D = torch.tensor(D, dtype=torch.float32).to(device)  # [N, T_in]
X = torch.cat((X, D.unsqueeze(1)), dim=1)  # [N, 4, T_in]

# print('Signal is attached')
# X = torch.cat((X, y.unsqueeze(1)), dim=1)  # [N, 5, T_in]

print('Signal is not attached')
X = torch.cat((X, torch.zeros_like(y).unsqueeze(1)), dim=1)  # [N, 5, T_in]

print(X.shape, y.shape)

N_in = X.shape[1]   # Number of input channels +1 (for distance gate)
N_out = 1  # Number of output channels (MS2 signal)
T_in = y.shape[1]
model_unet = UNet1DVariableDecoder(N_in, encoder_depth=1, decoder_depth=1, base_channels=8,
                                   init_thresh=0.15, init_alpha=100.0, learn_alpha=False,
                                   output_length=X.shape[-1],
                                   use_DistanceGate_mask=False)

model_unet.to(device)

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
criterion = nn.SmoothL1Loss()
criterion = NormalizedCutLoss(k=2)
criterion = EntropyLoss()

criterion = nn.MSELoss()

criterion = CalibratedGaussianNLL(
    lambda_var=1e-4,           # try 1e-5 to 1e-3
    target_logvar=None,        # or math.log(sigma0**2) if you know noise level
    lambda_prior=1e-4,         # if you set target_logvar
    clip_logvar=(-8, 6),
    warmup_steps=20,
    total_steps=100000
)

optimizer = torch.optim.Adam(
    list(model_unet.parameters()),
      lr=1e-3)


# ----- Training Loop -----
num_epochs = 50
train_losses = []
val_losses = []

best_val_loss = float('inf')  # initialize to a large value
for epoch in trange(num_epochs, desc="Training Progress"):
    train_loss, recon_loss, mask_loss, distance_loss = train_epoch(model_unet, train_loader, optimizer, criterion, device)
    val_loss, val_preds, val_targets, best_val_loss = validate_epoch(model_unet, val_loader, criterion, 
                                                      best_val_loss=best_val_loss, best_model_state=None,
                                                      device=device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

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
checkpoint = torch.load("best_model_EPMS2_to_states.pt", map_location='cpu')
model_unet.load_state_dict(checkpoint['model_unet'])

print("Learned threshold distance:", model_unet.gate.dc.item())
print("Learned alpha distance:", model_unet.gate.alpha.item())


gate = torch.sigmoid(model_unet.gate.alpha.item() * (model_unet.gate.dc.item() - X_test[:,3]))

X_test_gated = X_test[:,4] * gate

print(X_test_gated.shape)

idx_chosen = np.random.randint(0, len(X_val))

fig, ax = plt.subplots(2,1,figsize=(12, 6))
ax[0].plot(X_test_gated[idx_chosen].cpu().detach().numpy(), label='Gated MS2 Signal', color='C0')
ax[1].plot(X_test[idx_chosen,4].cpu().detach().numpy(), label='MS2 Signal', color='C0')
ax[0].set_xlabel("Time")
ax[0].set_ylabel("Gated MS2 Value")
ax[1].set_ylabel("MS2 Signal Value")
ax[0].legend()


model_unet.eval()
model_unet.to('cpu')

mean, logvar, distance_signal = model_unet(X_test.to('cpu'))  # [B, 1, T_out]
binary = torch.sigmoid(mean)  # Apply sigmoid to get probabilities
binary0 = binary.squeeze(1)  # [B, T_out]
output = mean.squeeze(1)  # [B, T_in]
std = torch.sqrt(torch.exp(logvar)).squeeze(1)  # [B, T_in]

print(output[0].shape, std[0].shape)
print(output[idx_chosen].cpu().detach().numpy()-std[idx_chosen].cpu().detach().numpy())
print(output[idx_chosen].cpu().detach().numpy()+std[idx_chosen].cpu().detach().numpy())

fig, ax = plt.subplots(3,1,figsize=(12, 9))
ax[0].plot(X_test[idx_chosen,3,:].cpu().numpy().flatten(), label='Signal', color='C0')
ax[0].plot(X_test[idx_chosen,3,:].cpu().numpy().flatten()<=0.1, label='Under radius', color='k')

ax[1].plot(y_test[idx_chosen, 0,:].cpu().numpy().flatten(), label='GT State', color='C1')
ax[1].plot(output[idx_chosen].cpu().detach().numpy().flatten(), label='Pred. State', color='C2')
ax[1].fill_between(np.arange(len(output[idx_chosen])), 
                   output[idx_chosen].cpu().detach().numpy()-std[idx_chosen].cpu().detach().numpy(), 
                   output[idx_chosen].cpu().detach().numpy()+std[idx_chosen].cpu().detach().numpy(),
                   alpha=0.2, color='C2'
                   )

ax[2].plot(output[idx_chosen].cpu().detach().numpy().flatten(), label='Pred. State', color='C2')

output_minmax = output.cpu().detach().numpy()

output_minmax = [(np.clip(o, 0, 1) - np.min(o)) / (np.max(o) - np.min(o)) for o in output_minmax]
output_minmax = np.array(output_minmax)


print(output_minmax.min())
print(output_minmax.max())
print(output_minmax.shape)

output_binary = output_minmax > 2*np.std(output_minmax)  # Apply sigmoid to get probabilities

print('.edoian', np.median(output_minmax)+np.std(output_minmax), np.median(output_minmax), np.std(output_minmax) )

x = torch.arange(0,1,0.01)
y = torch.sigmoid(model_unet.gate.alpha.item() * (model_unet.gate.dc.item() - x))
plt.figure(figsize=(5, 4))
plt.plot(x, y, label='Sigmoid Gate Function')
plt.annotate('Function: \ntorch.sigmoid(\n100 * (dc - distance)\n)', 
             xy=(0.45, 0.55))
plt.axvline(model_unet.gate.dc.item(), color='green', linestyle='--', label='Learned $d_c$: {:.4f} um \nTrue: 0.1 um'.format(model_unet.gate.dc.item()))
plt.xlabel('Distance')
plt.ylabel('Gate Output')
plt.title('Distance Gate Function')
plt.legend(loc='upper right')

print(torch.sigmoid(model_unet.gate.alpha.item() * (model_unet.gate.dc.item() - torch.tensor([0.05]))).item())
print(torch.sigmoid(model_unet.gate.alpha.item() * (model_unet.gate.dc.item() - torch.tensor([0.1]))).item())
print(torch.sigmoid(model_unet.gate.alpha.item() * (model_unet.gate.dc.item() - torch.tensor([0.2]))).item())
print(torch.sigmoid(model_unet.gate.alpha.item() * (model_unet.gate.dc.item() - torch.tensor([0.3]))).item())


print(f"Binary output shape: {binary0.shape}")
print(f"Output shape: {output.shape}")
print(f"X_test shape: {X_test.shape}")
# plot some outputs versus targets



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
axs[1].plot(y_test[idx_chosen, 0,:].cpu().numpy().flatten(), label='GT State', color='C1')
axs[1].plot(output[idx_chosen].cpu().detach().numpy().flatten(), label='Pred. State', color='C2')
axs[1].fill_between(np.arange(len(output[idx_chosen])), 
                   output[idx_chosen].cpu().detach().numpy()-std[idx_chosen].cpu().detach().numpy(), 
                   output[idx_chosen].cpu().detach().numpy()+std[idx_chosen].cpu().detach().numpy(),
                   alpha=0.2, color='C2'
                   )
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Signal Value")
axs[1].set_title(f"Residual of Output vs Binary sequence for X_test {idx_chosen+1}")
axs[1].legend()

axs[2].plot(X_test[idx_chosen,3,:].cpu().numpy().flatten()<=0.15, label='Under 0.15', color='k')
# twin axs
axtwin = axs[2].twinx()
axtwin.plot(output[idx_chosen].cpu().detach().numpy().flatten(), label='Pred. State', color='C2')
axtwin.fill_between(
                   np.arange(len(output[idx_chosen])), 
                   output[idx_chosen].cpu().detach().numpy()-std[idx_chosen].cpu().detach().numpy(), 
                   output[idx_chosen].cpu().detach().numpy()+std[idx_chosen].cpu().detach().numpy(),
                   alpha=0.2, color='C2'
                   )
axtwin.set_ylim(0, np.max(output[idx_chosen].cpu().detach().numpy()+std[idx_chosen].cpu().detach().numpy()))

axs[2].set_xlabel("Time")
axs[2].set_ylabel("Signal Value")
axs[2].set_title(f"Residual of Output vs Binary sequence for X_test {idx_chosen+1}")
axs[2].legend()

axs[3].plot(y_test[idx_chosen, 0,:].cpu().numpy().flatten(), label='GT State', color='C1')
axs[3].plot(output[idx_chosen].cpu().detach().numpy().flatten(), label='Pred. State', color='C2')

axs[3].plot(X_test_gated[idx_chosen].cpu().detach().numpy(), label='Gated Signal', color='C3')

#axs[3].plot(, label='Pred. State', color='C2')

axs[3].set_xlabel("Time")
axs[3].set_ylabel("Signal Value")
axs[3].set_title(f"Residual of Output vs Binary sequence for X_test {idx_chosen+1}")
axs[3].legend()

# axs[4].plot(y_test[idx_chosen, 1,:].cpu().numpy().flatten(), label='GT State', color='C1')
# axs[4].plot(output_binary[idx_chosen], label='Pred. State (Min-Max Norm)', color='C3')
# axs[4].set_xlabel("Time")
# axs[4].set_ylabel("Signal Value")
# axs[4].set_title(f"Residual of Output vs Binary sequence for X_test {idx_chosen+1}")
# axs[4].legend()


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
    inarray = output_binary[i]
    segment_lengths, segment_starts, segment_types = find_segments(inarray)
    output_binary_seglengths.append(segment_lengths)
output_binary_seglengths_flat = np.concatenate(output_binary_seglengths)
    

plt.figure(figsize=(5, 4))
plt.hist(states_all_test_seglengths_flat, bins=100, alpha=0.5, density=True, label='GT State Segments')
plt.hist(output_binary_seglengths_flat, bins=100, alpha=0.5, density=True, label='Pred. Binary Segments')
plt.xlabel('Segment Length')
plt.ylabel('Frequency')
plt.title('Segment Length Distribution')
plt.legend(loc='upper right')

plt.annotate(f'Avg. seg. length GT: {np.mean(states_all_test_seglengths_flat):.2f}\n'
             f'Avg. seg. length Pred: {np.mean(output_binary_seglengths_flat):.2f}',
             xy=(0.5, 0.8), xycoords='axes fraction',
             fontsize=10, ha='left', va='top')
plt.show()


# Acc metrics for each sequence

Acc_all = []
for i in range(len(output_binary)):
    acc = (output_binary[i] == states_all_test[i].cpu().numpy().flatten()).mean() 
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

# pip install captum
import torch
import torch.nn as nn
from typing import Optional, Sequence, Union

from captum.attr import (
    IntegratedGradients,
    Saliency,
    Occlusion,
    DeepLift,
    LRP,
)

"""

Targets: Because wrapped(x) returns [B, T], target is the time index whose contribution you want. Pass a single int or a list to aggregate (sum) across indices.
Baselines: For IG/DeepLIFT, start with zeros. If your inputs are standardized, a mean input baseline can be better.
Occlusion: The time_window you choose should roughly match the temporal footprint you want to test (e.g., a kernel-receptive-field scale).
Batching: All functions accept [B, C, T] and return attributions of the same shape (aggregated over chosen t_idx if a list).
Stability: If you see noisy saliency, try smoothing (average attributions over small neighborhoods of t_idx) or use IG with more n_steps.

"""

############################################
# 1) Wrapper so Captum sees ŷ_mean[B, T]
############################################
class MeanHeadWrapper(nn.Module):
    """
    Wraps your probabilistic model to expose just the mean output
    with shape [B, T], suitable for Captum targets (time indices).
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Your model returns (mean [B,1,T], logvar [B,1,T], distance_signal)
        mean, _, _ = self.model(x)
        return mean.squeeze(1)  # [B, T]


############################################
# 2) Baselines & utilities
############################################
def make_baseline(
    x: torch.Tensor,
    kind: str = "zeros",
    value: float = 0.0
) -> torch.Tensor:
    """
    Create a baseline tensor for attribution methods that need one.
    kind: "zeros" | "value" | "noise"
    """
    if kind == "zeros":
        return torch.zeros_like(x)
    elif kind == "value":
        return torch.full_like(x, fill_value=value)
    elif kind == "noise":
        return torch.randn_like(x) * 0.01
    else:
        raise ValueError(f"Unknown baseline kind: {kind}")


def _ensure_eval_and_requires_grad(model: nn.Module, x: torch.Tensor):
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    return x


############################################
# 3) Integrated Gradients
############################################
@torch.no_grad()
def integrated_gradients(
    wrapped: nn.Module,
    x: torch.Tensor,
    t_idx: Union[int, Sequence[int]],
    n_steps: int = 128,
    baseline_kind: str = "zeros",
    baseline_value: float = 0.0
) -> torch.Tensor:
    """
    Returns IG attributions with shape [B, C, T] for a single target index or
    a list of indices aggregated by sum across targets.
    """
    x = _ensure_eval_and_requires_grad(wrapped, x)
    baseline = make_baseline(x, baseline_kind, baseline_value)
    ig = IntegratedGradients(wrapped)

    if isinstance(t_idx, Sequence):
        attrs = []
        for t in t_idx:
            a = ig.attribute(inputs=x, baselines=baseline, target=t, n_steps=n_steps)
            attrs.append(a)
        return torch.stack(attrs, dim=0).sum(0)  # sum across selected targets → [B,C,T]
    else:
        return ig.attribute(inputs=x, baselines=baseline, target=t_idx, n_steps=n_steps)


############################################
# 4) Saliency (vanilla gradients)
############################################
@torch.no_grad()
def saliency_maps(
    wrapped: nn.Module,
    x: torch.Tensor,
    t_idx: Union[int, Sequence[int]],
    absolute: bool = True
) -> torch.Tensor:
    """
    Returns saliency attributions [B,C,T]. If multiple t_idx are passed,
    sums attributions over those targets.
    """
    x = _ensure_eval_and_requires_grad(wrapped, x)
    sal = Saliency(wrapped)

    if isinstance(t_idx, Sequence):
        attrs = []
        for t in t_idx:
            a = sal.attribute(inputs=x, target=t, abs=absolute)
            attrs.append(a)
        A = torch.stack(attrs, dim=0).sum(0)
        return A.abs() if absolute else A
    else:
        a = sal.attribute(inputs=x, target=t_idx, abs=absolute)
        return a


############################################
# 5) Occlusion (sliding window along time)
############################################
@torch.no_grad()
def occlusion_sensitivity(
    wrapped: nn.Module,
    x: torch.Tensor,
    t_idx: Union[int, Sequence[int]],
    time_window: int = 21,
    stride: int = 7,
    perturbation: float = 0.0,
) -> torch.Tensor:
    """
    Occlusion with a temporal window. Uses a full-channel × time_window patch.
    Returns attributions [B,C,T].
    """
    x = _ensure_eval_and_requires_grad(wrapped, x)
    occ = Occlusion(wrapped)

    # For input of shape [B, C, T], sliding_window_shapes excludes batch dim → (C, time_window)
    sws = (x.shape[1], time_window)
    st = (1, stride)

    if isinstance(t_idx, Sequence):
        attrs = []
        for t in t_idx:
            a = occ.attribute(
                inputs=x,
                target=t,
                sliding_window_shapes=sws,
                strides=st,
                baselines=perturbation,  # a scalar baseline fills the occluded window
            )
            attrs.append(a)
        return torch.stack(attrs, dim=0).sum(0)
    else:
        return occ.attribute(
            inputs=x,
            target=t_idx,
            sliding_window_shapes=sws,
            strides=st,
            baselines=perturbation,
        )


############################################
# 6) DeepLIFT
############################################
@torch.no_grad()
def deeplift_attr(
    wrapped: nn.Module,
    x: torch.Tensor,
    t_idx: Union[int, Sequence[int]],
    baseline_kind: str = "zeros",
    baseline_value: float = 0.0
) -> torch.Tensor:
    """
    DeepLIFT attributions [B,C,T]. Requires a baseline.
    """
    x = _ensure_eval_and_requires_grad(wrapped, x)
    baseline = make_baseline(x, baseline_kind, baseline_value)
    dl = DeepLift(wrapped)

    if isinstance(t_idx, Sequence):
        attrs = []
        for t in t_idx:
            a = dl.attribute(inputs=x, baselines=baseline, target=t)
            attrs.append(a)
        return torch.stack(attrs, dim=0).sum(0)
    else:
        return dl.attribute(inputs=x, baselines=baseline, target=t_idx)


############################################
# 7) LRP
############################################
@torch.no_grad()
def lrp_attr(
    wrapped: nn.Module,
    x: torch.Tensor,
    t_idx: Union[int, Sequence[int]],
) -> torch.Tensor:
    """
    LRP attributions [B,C,T].
    Note: Captum's LRP works best with ReLU-style nets; convs are supported.
    """
    x = _ensure_eval_and_requires_grad(wrapped, x)
    lrp = LRP(wrapped)

    if isinstance(t_idx, Sequence):
        attrs = []
        for t in t_idx:
            a = lrp.attribute(inputs=x, target=t)
            attrs.append(a)
        return torch.stack(attrs, dim=0).sum(0)
    else:
        return lrp.attribute(inputs=x, target=t_idx)


############################################
# 8) Example usage
############################################
model = model_unet
wrapped = MeanHeadWrapper(model)     # expose mean[B,T]
#x: torch.Tensor of shape [B, C, T] on the same device as model
x = X_test.to(next(model.parameters()).device)

# Choose a specific time index, or a list (e.g., a segment)
t_idx = 300
t_idx = [295, 296, 305]

# IG:
ig_attr = integrated_gradients(wrapped, x, t_idx, n_steps=128, baseline_kind="zeros")

# Saliency:
sal_attr = saliency_maps(wrapped, x, t_idx, absolute=True)

# Occlusion:
occ_attr = occlusion_sensitivity(wrapped, x, t_idx, time_window=25, stride=5, perturbation=0.0)

# DeepLIFT:
dl_attr = deeplift_attr(wrapped, x, t_idx, baseline_kind="zeros")

# LRP:
lrp_attr = lrp_attr(wrapped, x, t_idx)

############################################
# 9) Optional: normalize for visualization
############################################
def normalize_attr(a: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize per-sample to [-1, 1] by max absolute value.
    Keeps sign information.
    """
    a = a.detach()
    B = a.shape[0]
    a = a / (a.view(B, -1).abs().amax(dim=1).view(B, 1, 1) + eps)
    return a



import matplotlib.pyplot as plt
import numpy as np

def plot_attributions(
    attributions: torch.Tensor,
    inputs: Optional[torch.Tensor] = None,
    sample_idx: int = 0,
    channel_names: Optional[Sequence[str]] = None,
    cmap: str = "seismic",
    normalize: bool = True,
    figsize: tuple = (12, 2.5),
):
    """
    Plot per-channel per-time attributions for one sample.

    Parameters
    ----------
    attributions : torch.Tensor
        Shape [B, C, T]. Attribution scores.
    inputs : torch.Tensor, optional
        Shape [B, C, T]. Original inputs to overlay (scaled).
    sample_idx : int
        Which batch element to plot.
    channel_names : list of str, optional
        Names for each channel (default: "Ch0", "Ch1", ...).
    cmap : str
        Matplotlib colormap for attribution heatmaps.
    normalize : bool
        Whether to normalize attribution values to [-1,1] per channel.
    figsize : tuple
        Figure size per channel.
    """
    A = attributions[sample_idx].detach().cpu().numpy()  # [C,T]
    if inputs is not None:
        X = inputs[sample_idx].detach().cpu().numpy()    # [C,T]
    else:
        X = None

    C, T = A.shape
    if channel_names is None:
        channel_names = [f"Ch{i}" for i in range(C)]

    fig, axes = plt.subplots(C, 1, figsize=(figsize[0], figsize[1] * C), sharex=True)

    if C == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        a = A[i]
        if normalize:
            max_abs = np.max(np.abs(a)) + 1e-8
            a = a / max_abs
        im = ax.imshow(
            a[np.newaxis, :], aspect="auto", cmap=cmap,
            extent=[0, T, -0.5, 0.5], vmin=-1, vmax=1
        )
        ax.set_yticks([])
        ax.set_ylabel(channel_names[i], rotation=0, labelpad=30, va="center")

        if X is not None:
            sig = X[i]
            sig_norm = sig / (np.max(np.abs(sig)) + 1e-8)
            ax.plot(np.linspace(0, T, len(sig)), sig_norm * 0.5, color="black", alpha=0.7)

    axes[-1].set_xlabel("Time")
    fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.01, pad=0.02, label="Attribution")
    plt.tight_layout()
    plt.show()


# Example: integrated gradients attribution
ig_attr = integrated_gradients(wrapped, x, t_idx=300, n_steps=128)

# Plot for the first sample
plot_attributions(ig_attr, inputs=x, sample_idx=0,
                  channel_names=["Signal", "Feat1", "Feat2", "Distance"])