# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from tqdm import trange
from torch.utils.data import Dataset, DataLoader

from helper_functions_toyUnet import *

### Variables ###

# model variables
batch_size = 64
num_epochs = 100
patience = 20
lr = 1e-4

# Generate data
use_MS2_data = False
attach_MS2 = False

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    X = X.permute(0, 2, 1)
else:
    torch.manual_seed(0)
    np.random.seed(0)
    X, y = generate_data(n_samples=10000)
    y = y.permute(0, 2, 1)#.squeeze(-1).long()




fig, ax = plt.subplots(2,1,figsize=(10, 4))
ax[0].plot(X[0,:,3].cpu().numpy(), label='feat 0')
ax[0].legend()
ax[0].set_title("Input Features")
ax[0].set_xlabel("Time Step")
ax[0].set_ylabel("Value")

print(y.shape)
ax[1].plot(y[0,0,:].cpu().numpy(), label='target', color='black')
ax[1].legend()
ax[1].set_title("Target")
ax[1].set_xlabel("Time Step")
ax[1].set_ylabel("Value")

# %%

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Class imbalance weights - if doing classification
# pos_weight_value = ((y_train == 0).sum() / (y_train == 1).sum()).item()
# weights = torch.tensor([1-pos_weight_value, pos_weight_value], device=device)

# Move to device
X_idx = np.arange(len(X))
X_train_idx, X_test_idx = train_test_split(X_idx, test_size=0.2, random_state=42)
X_train_idx, X_val_idx = train_test_split(X_train_idx, test_size=0.25, random_state=42)

X_train = X[X_train_idx].to(device)  # [N_train, 1, T_in]
X_val = X[X_val_idx].to(device)      # [N_val, 1, T_in]
X_test = X[X_test_idx].to(device)    # [N_test, 1, T_in]

y_train = y[X_train_idx].to(device)  # [N_train, T_in]
y_val = y[X_val_idx].to(device)      # [N_val, T_in]
y_test = y[X_test_idx].to(device)    # [N_test, T_in

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset   = TimeSeriesDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Training samples: {X_train.shape}, Validation samples: {X_val.shape}")
print(f"Training y: {y_train.shape}, Validation y: {y_val.shape}, unique. y {np.unique(y_train.cpu().numpy(), return_counts=True)}")


# ----- Model, Loss, Optimizer -----

# model = TransformerModel(vocab_size=X.shape[-1],
#                          d_model=6, 
#                          nhead=2, 
#                          num_encoder_layers=1,
#                          num_decoder_layers=1, 
#                          dim_feedforward=2, 
#                          dropout=0.01, 
#                          max_len=1000
#                          ).to(device)

model = UNet1DVariableDecoder_resnet(
                              X.shape[-1], 
                              encoder_depth=3, decoder_depth=3, base_channels=64,
                              init_thresh=0.15, init_alpha=100.0, learn_alpha=False,
                              output_length=X.shape[1],
                              use_DistanceGate_mask=False).to(device)

# Optimizer
optimizer = torch.optim.Adam(
    list(model.parameters()),
      lr=lr)

# Criterion

# criterion = nn.CrossEntropyLoss(weight=weights)
# criterion = GaussianNLLLoss()
criterion = nn.MSELoss()
criterion = CalibratedGaussianNLL(
    lambda_var=1e-4,           # try 1e-5 to 1e-3
    target_logvar=None,        # or math.log(sigma0**2) if you know noise level
    lambda_prior=1e-4,         # if you set target_logvar
    clip_logvar=(-8, 6),
    warmup_steps=20,
    total_steps=100000
)

# ----- Training Loop with Validation -----

train_losses = []
val_losses = []

patience_counter = 0
best_val_loss = np.inf
for epoch in trange(num_epochs, desc="Training"):
    
    # ----- Training -----
    model.train()
    batch_train_losses = []
    for i, (X_batch, y_batch) in enumerate(train_loader):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        output, output_logvar, distance_signal = model(X_batch)        
        output = output.clamp(-20, 20)

        # Things need same shape for correct loss computations
        if y_batch.dim() == 3:
            y_batch = y_batch.squeeze(1)
        if output.dim() == 3:
            output = output.squeeze(1)
        if output_logvar.dim() == 3:
            output_logvar = output_logvar.squeeze(1)
        assert y_batch.shape == output.shape == output_logvar.shape, f'Shape mismatch: {y_batch.shape}, {output.shape}, {output_logvar.shape}'

        # Compute loss - with catch statement for MSE vs Calibrated NLL
        try:
            loss = criterion(output, y_batch)  # regression
        except:
            loss = criterion(output, output_logvar, y_batch).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        batch_train_losses.append(loss.item())
    train_losses.append(sum(batch_train_losses) / len(batch_train_losses))

    # ----- Validation -----
    model.eval()
    batch_val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            val_output, val_output_logvar, distance_signal = model(X_batch)
            val_output = val_output.clamp(-20, 20)

            if y_batch.dim() == 3:
                y_batch = y_batch.squeeze(1)
            if val_output.dim() == 3:
                val_output = val_output.squeeze(1)
            if val_output_logvar.dim() == 3:
                val_output_logvar = val_output_logvar.squeeze(1)

            # Things need same shape for correct loss computations
            assert y_batch.shape == val_output.shape == val_output_logvar.shape

            try:
                val_loss = criterion(val_output, y_batch)
            except:
                val_loss = criterion(val_output, val_output_logvar, y_batch).mean()
            batch_val_losses.append(val_loss.item())

    avg_val_loss = sum(batch_val_losses) / len(batch_val_losses)
    val_losses.append(avg_val_loss)

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
        pickle.dump(best_model_state, open('toy_transformer_model.pickle', 'wb'))
        print('\tNew best model, validation loss {}'.format(best_val_loss))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if epoch % 1 == 0:
        print(f"Epoch {epoch}: Train Loss={train_losses[-1]:.6f}, Val Loss={avg_val_loss:.6f}")

pickle.dump(train_losses, open('toy_transformer_trainloss.pickle', 'wb'))
pickle.dump(val_losses, open('toy_transformer_valloss.pickle', 'wb'))


