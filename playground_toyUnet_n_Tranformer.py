# %%
import torch
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
from torch.utils.data import Dataset, DataLoader

from helper_functions_toyUnet import *

# CUDA_VISIBLE_DEVICES=0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Generate data
use_MS2_data = True
attach_MS2 = False


if use_MS2_data:
    C, D, y, fulltime_padded_states_all = Load_simulated_data(path='_data/dataset_for_Jacob.pkl')
    y = torch.stack(y).to(device)
    X = torch.stack(C).to(device)  # [N, 3, T_in] 
    D = torch.tensor(D, dtype=torch.float32).to(device)  # [N, T_in]
    X = torch.cat((X, D.unsqueeze(1)), dim=1)  # [N, 4, T_in]
    if attach_MS2:
        X = torch.cat((X, y.unsqueeze(1)), dim=1)  # [N, 5, T_in]
    else:
        X = torch.cat((X, torch.zeros_like(y).unsqueeze(1)), dim=1)  # [N, 5, T_in]
    # permute to [N, T_in, 5]
    X = X.permute(0, 2, 1)
    y = y.unsqueeze(-1)
else:
    X, y = generate_data(n_samples=600)
    y = y#.squeeze(-1).long()


# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Class imbalance weights
pos_weight_value = ((y_train == 0).sum() / (y_train == 1).sum()).item()
weights = torch.tensor([0.01, 1.0], device=device)

# Move to device
X_train, y_train = X_train.to(device), y_train.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)

batch_size = 32

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset   = TimeSeriesDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


print(f"Training samples: {X_train.shape}, Validation samples: {X_val.shape}")
print(f"Training y: {y_train.shape}, Validation y: {y_val.shape}, unique. y {np.unique(y_train.cpu().numpy(), return_counts=True)}")


# ----- Model, Loss, Optimizer -----
model = TransformerModel(vocab_size=X.shape[-1],
                         d_model=32, 
                         nhead=4, 
                         num_encoder_layers=4,
                         num_decoder_layers=4, 
                         dim_feedforward=6, 
                         dropout=0.1, 
                         max_len=1000
                         ).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(weight=weights)
criterion = nn.MSELoss()

# ----- Training Loop with Validation -----
num_epochs = 10
patience = 5

train_losses = []
val_losses = []

patience_counter = 0
best_val_loss = np.inf
for epoch in trange(num_epochs, desc="Training"):
    model.train()
    batch_train_losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output = model(X_batch)             # [B, T, 1]
        loss = criterion(output, y_batch)  # regression
        loss.backward()
        optimizer.step()
        batch_train_losses.append(loss.item())

    train_losses.append(sum(batch_train_losses) / len(batch_train_losses))

    # ----- Validation -----
    model.eval()
    batch_val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_output = model(X_batch)
            val_loss = criterion(val_output, y_batch)
            batch_val_losses.append(val_loss.item())

    avg_val_loss = sum(batch_val_losses) / len(batch_val_losses)
    val_losses.append(avg_val_loss)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 1 == 0:
        print(f"Epoch {epoch}: Train Loss={train_losses[-1]:.6f}, Val Loss={avg_val_loss:.6f}")

plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()


# Restore best model
model.load_state_dict(best_model_state)

# ----- Evaluation + Visualization -----
idx_chosen = np.random.randint(0, X_val.shape[0])

model.eval()
input_ts = X_val[idx_chosen:idx_chosen+1].clone().detach().requires_grad_()
pred = model(input_ts)

print(pred, X_val[idx_chosen:idx_chosen+1])
probs = torch.softmax(pred, dim=-1)
pred_timestep = torch.argmax(probs.squeeze()).item()

fig, ax = plt.subplots(4,1,figsize=(10, 6))
ax[0].plot(input_ts.squeeze().detach().cpu(), label='Input Time Series')
ax[0].legend()
ax[0].set_title("Input Time Series")

print(pred.squeeze().detach().cpu())

ax[1].plot(y_val[idx_chosen].squeeze().cpu(), label='Target')
ax[1].plot(pred.squeeze().detach().cpu(), label='Predicted Probabilities')
ax[1].set_title("Target vs Predicted Probabilities")
ax[1].legend()
ax[1].set_ylim(y_val[idx_chosen].squeeze().cpu().min()-1, y_val[idx_chosen].squeeze().cpu().max()+1)


# ----- Saliency -----
class TimeStepClassifierWrapper(nn.Module):
    def __init__(self, model, timestep, class_idx):
        super().__init__()
        self.model = model
        self.timestep = timestep
        self.class_idx = class_idx

    def forward(self, x):
        logits = self.model(x)
        return logits[:, self.timestep, self.class_idx]

wrapped_model = TimeStepClassifierWrapper(model, pred_timestep, class_idx=1)
saliency = Saliency(wrapped_model)
attr_sal = saliency.attribute(input_ts)
attr_sal = attr_sal.abs().detach().squeeze().cpu().numpy()
ax[2].plot(attr_sal, label='Saliency')
ax[2].set_title("Saliency Attribution")
ax[2].legend()


# ----- Integrated Gradients -----
ig = IntegratedGradients(wrapped_model)
baseline = torch.zeros_like(input_ts)
attr_ig = ig.attribute(input_ts, baselines=baseline)
attr_ig = attr_ig.detach().squeeze().cpu().numpy()
ax[3].plot(attr_ig, label='Integrated Gradients')
ax[3].plot(np.abs(attr_ig), label='Abs. Integrated Gradients')
ax[3].set_title("Integrated Gradients Attribution")
ax[3].legend()

plt.tight_layout()

# ----- Attention -----
with torch.no_grad():
    _ = model(input_ts)

# Plot attention for each layer and head
for layer_idx, layer_attn in enumerate(model.attn_weights_all_layers):
    for head_idx in range(layer_attn.shape[1]):
        attn_map = layer_attn[0, head_idx].cpu()  # [T, T]
        plt.figure(figsize=(6, 5))
        sns.heatmap(attn_map, cmap='viridis')
        plt.title(f"Layer {layer_idx} - Head {head_idx} Attention")
        plt.xlabel("Source Timestep")
        plt.ylabel("Target Timestep")
        plt.tight_layout()
        plt.show()


# -------- Additional Attention Visualizations --------
def plot_average_attention(attn_weights):
    for l, attn in enumerate(attn_weights):
        for h in range(attn.shape[1]):
            avg_attn = attn[0, h].mean(dim=0).cpu().numpy()
            plt.figure()
            sns.heatmap(avg_attn[None, :], cmap='viridis', cbar=True)
            plt.title(f'Layer {l+1}, Head {h+1} - Avg Attention')
            plt.xlabel("Source Timestep")
            plt.yticks([])
            plt.tight_layout()

plot_average_attention(model.attn_weights_all_layers)


def compute_rollout(attn_weights):
    rollout = torch.eye(attn_weights[0].shape[-1]).to(attn_weights[0].device)
    for attn in attn_weights:
        attn_head_avg = attn.mean(dim=1)  # [B, T, T]
        rollout = attn_head_avg[0] @ rollout
    return rollout.cpu().numpy()

def plot_attention_rollout(rollout_matrix):
    plt.figure(figsize=(6, 5))
    sns.heatmap(rollout_matrix, cmap="viridis")
    plt.title("Attention Rollout")
    plt.xlabel("Source timestep")
    plt.ylabel("Target timestep")
    plt.tight_layout()

rollout_matrix = compute_rollout(model.attn_weights_all_layers)
plot_attention_rollout(rollout_matrix)


def plot_attention_entropy(attn_weights):
    for l, attn in enumerate(attn_weights):
        for h in range(attn.shape[1]):
            entropy = - (attn[0, h] * torch.log(attn[0, h] + 1e-8)).sum(dim=-1).cpu().numpy()
            plt.plot(entropy, label=f"Layer {l+1} Head {h+1}")
    plt.title("Attention Entropy per Head")
    plt.xlabel("Target Timestep")
    plt.ylabel("Entropy")
    plt.legend()
    plt.tight_layout()

plot_attention_entropy(model.attn_weights_all_layers)


def plot_attention_at_timestep(attn_weights, timestep):
    for l, attn in enumerate(attn_weights):
        for h in range(attn.shape[1]):
            plt.figure()
            sns.heatmap(attn[0, h][timestep].cpu().numpy()[None, :], cmap="viridis")
            plt.title(f"Layer {l+1}, Head {h+1} - Attention at Timestep {timestep}")
            plt.xlabel("Source Timestep")
            plt.tight_layout()

plot_attention_at_timestep(model.attn_weights_all_layers, pred_timestep)


def plot_aggregated_attention(attn_weights):
    stacked = torch.stack([attn[0].mean(dim=0) for attn in attn_weights], dim=0)  # [L, T, T]
    avg_attn = stacked.mean(dim=0).cpu().numpy()
    plt.figure(figsize=(6, 5))
    sns.heatmap(avg_attn, cmap='viridis')
    plt.title("Aggregated Attention (Layer+Head Avg)")
    plt.xlabel("Source")
    plt.ylabel("Target")
    plt.tight_layout()

plot_aggregated_attention(model.attn_weights_all_layers)


# Optional: Compare saliency with attention
sal = attr_sal.squeeze() / np.abs(attr_sal).max()
attn = model.attn_weights_all_layers[0][0].mean(dim=0)[pred_timestep].cpu().numpy() / np.abs(attr_sal).max()

plt.figure(figsize=(10, 4))
plt.plot(sal, label="Saliency")
plt.plot(attn, label=f"Attention @ t={pred_timestep}")
plt.legend()
plt.title("Saliency vs Attention")
plt.xlabel("Timestep")
plt.tight_layout()

