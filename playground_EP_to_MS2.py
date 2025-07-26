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
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    def __init__(self, num_features=1, d_model=6, nhead=1, num_layers=1):
        """
        Transformer model for time series classification/regression.

        Parameters
        ----------
        num_features : int
            Number of input features per time step.
        d_model : int
            Dimension of the model (embedding size).
        nhead : int
            Number of attention heads.
        num_layers : int
            Number of transformer encoder layers.
        """

        super().__init__()
        self.embedding = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, 1)

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
    
    def forward_lambda(self, x):
        # x: EP separation observations (batchsize, seqlength, xyz) + 
        
        # transformer or Unet der encoder og decoder men end result er større for allow more context og convolution til seqlen
        x_embed = network(x) # x: (batchsize, seqlength, xyz) -> x_embed (batchsize, seqlength + W, 1) 
        x_embed = relu(x_embed) # Apply ReLU activation
        
        # Apply single convolutional layer
        y = cnn(x_embed) # x_embed: (batchsize, seqlength + W, 1) -> x_embed: (batchsize, seqlength, 1)

        
        return y


# ----- Data Generation -----
def generate_data(n_samples=1000, seq_len=50):
    X = torch.empty(n_samples, seq_len, 1)
    y = torch.zeros(n_samples, seq_len, 1)
    for i in range(n_samples):
        X[i] = torch.cumsum(torch.randn(seq_len, 1), dim=0)
        random_idx = torch.randint(5, seq_len-5, (1,))
        start_idx = max(random_idx - 3, 0)
        end_idx = min(random_idx + 1, seq_len)
        X[i, start_idx:end_idx] += 10.0
        y[i, random_idx] = 1.0
    return X, y

torch.manual_seed(42)

import pickle
with open('data/dataset_for_Jacob.pkl', 'rb') as f:
    data = pickle.load(f)

C = []
D = []
y = []
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

print(f"Number of samples: {len(C)}")
print(f"Shape of C: {C[0].shape}, Shape of D: {len(D[0])}, Shape of y: {y[0].shape}")

# coords C and distances D
plt.figure(figsize=(10, 4))
plt.plot(D[0])
plt.title("Distance between EPs")
plt.xlabel("Time")
plt.ylabel("Distance")
plt.show()


C = torch.stack(C).permute(0, 2, 1)
D = torch.tensor(D, dtype=torch.float32).unsqueeze(-1)


X = D
X = torch.cat((C, D), dim=-1)  # Concatenate distance as a feature

y = torch.stack(y)

print(X.shape, y.shape)

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Class imbalance weights
pos_weight_value = ((y_train == 0).sum() / (y_train == 1).sum()).item()
weights = torch.tensor([0.01, 1.0], device=device)

# Create Datasets
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ----- Model, Loss, Optimizer -----
model = Transformer(num_features=X_train.shape[-1]).to(device)
#criterion = nn.CrossEntropyLoss(weight=weights)
criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ----- Training Loop with Validation -----
best_val_f1 = 0.0
best_val_precision = 0.0
best_val_loss = float('inf')
patience = 5
patience_counter = 0

val_losses = []
train_losses = []

TP_history = []
TN_history = []
FP_history = []
FN_history = []

precision_history = []
recall_history = []

for epoch in trange(5, desc="Training"):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output.squeeze(-1), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_losses.append(epoch_loss / len(train_loader))

    # ----- Validation -----
    model.eval()
    val_preds = []
    val_targets = []
    val_loss_total = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_output = model(X_batch)
            val_loss_total += criterion(val_output.squeeze(-1), y_batch).item()
            val_preds.append(torch.sigmoid(val_output).squeeze(-1).cpu())
            val_targets.append(y_batch.cpu())

    val_preds = torch.cat(val_preds).numpy().flatten()
    val_targets = torch.cat(val_targets).numpy().flatten()
    val_loss = val_loss_total / len(val_loader)

    TP_count = ((val_preds == 1) & (val_targets == 1)).sum() 
    TN_count = ((val_preds == 0) & (val_targets == 0)).sum() 
    FP_count = ((val_preds == 1) & (val_targets == 0)).sum() 
    FN_count = ((val_preds == 0) & (val_targets == 1)).sum() 

    TP = TP_count / (TP_count + FN_count) if (TP_count + FN_count) > 0 else 0
    TN = TN_count / (TN_count + FP_count) if (TN_count + FP_count) > 0 else 0
    FP = FP_count / (FP_count + TN_count) if (FP_count + TN_count) > 0 else 0
    FN = FN_count / (FN_count + TP_count) if (FN_count + TP_count) > 0 else 0

    TP_history.append(TP)
    TN_history.append(TN)
    FP_history.append(FP)
    FN_history.append(FN)

    precision = TP_count / (TP_count + FP_count) if (TP_count + FP_count) > 0 else 0
    precision_history.append(precision)

    recall = TP_count / (TP_count + FN_count) if (TP_count + FN_count) > 0 else 0
    recall_history.append(recall)

    # ----- Early stopping -----
    #if f1 > best_val_f1:
    if val_loss < best_val_loss:
        best_val_precision = precision
        patience_counter = 0
        best_model_state = model.state_dict()

    print(f"Epoch {epoch+1}/{200}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}, Precision: {precision:.4f}")


# %%

plt.figure(figsize=(10, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(TP_history, label='TP')
plt.plot(TN_history, label='TN')
plt.plot(FP_history, label='FP')
plt.plot(FN_history, label='FN')
plt.xlabel('Epoch')
plt.ylabel('Rate')
plt.title('TP, TN, FP, FN Rates')
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(precision_history, label='Precision')
plt.plot(recall_history, label='Recall')
plt.xlabel('Epoch')
plt.ylabel('Rate')
plt.title('Precision and Recall')
plt.legend()
plt.show()

# Restore best model
model.load_state_dict(best_model_state)

# ----- Evaluation + Visualization -----
"""
Attention & Saliency Visualizations — What They Show

1. **Per-Head Per-Layer Attention Heatmaps**
   - Shows attention weights for each head in each transformer layer.
   - Each heatmap is a [T x T] matrix, where (i, j) indicates how much the representation at timestep i attends to timestep j.
   - Y-axis: target timestep (where attention is applied).
   - X-axis: source timestep (what is attended to).

2. **Timestep-Specific Attention**
   - Filters the attention heatmaps to visualize only the target timestep (typically the predicted class-1 timestep).
   - Reveals which timesteps most influence that prediction.

3. **Average Attention**
   - Averages attention weights across heads (or layers) to get a global picture.
   - Highlights general attention patterns and key contributing regions.

4. **Attention Rollout**
   - Multiplies attention matrices across layers to estimate the full flow of information from input to output.
   - Useful for understanding indirect influence paths across layers.

5. **Attention Entropy**
   - Computes the entropy of attention distributions per head.
   - Low entropy: focused attention (sharp peaks).
   - High entropy: diffuse attention (spread out across timesteps).

6. **Saliency Maps**
   - Gradient-based importance visualization.
   - Shows which input values most affect the output at a given timestep (e.g., prediction point).

7. **Integrated Gradients**
   - Attribution technique comparing the model’s prediction to a baseline (e.g., zero input).
   - Produces smoother, more reliable attribution scores than raw saliency.

These visualizations help interpret and debug how the Transformer model makes decisions over time in sequence classification tasks — especially in the context of rare events.
"""


model.eval()
input_ts = X_val[0:1].clone().requires_grad_()
input_ts = input_ts.to(device)
pred = model(input_ts)
print(criterion(pred.squeeze(-1), y_val[0].to(device)))
#probs = torch.softmax(pred, dim=-1)[..., 1]

pred_timestep = 80

fig, ax = plt.subplots(4,1,figsize=(10, 6))
ax[0].plot(input_ts.squeeze().detach().cpu(), label='Input Time Series')
ax[0].legend()
ax[0].set_title("Input Time Series")

ax[1].plot(y_val[0].squeeze().cpu(), label='Target')
ax[1].plot(pred.squeeze().detach().cpu(), label='Predicted Probabilities')
ax[1].set_title("Target vs Predicted Probabilities")
ax[1].legend()


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
sal = attr_sal.squeeze()
attn = model.attn_weights_all_layers[0][0].mean(dim=0)[pred_timestep].cpu().numpy()

plt.figure(figsize=(10, 4))
plt.plot(sal, label="Saliency")
plt.plot(attn, label=f"Attention @ t={pred_timestep}")
plt.legend()
plt.title("Saliency vs Attention")
plt.xlabel("Timestep")
plt.tight_layout()