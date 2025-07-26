# train.py
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from model import TimeSeriesTransformerWithHF
import numpy as np
import pickle

# ----------------- Load Dataset -----------------
def collate_fn(batch):
    X, y = zip(*batch)
    lengths = [x.size(0) for x in X]
    padded = torch.nn.utils.rnn.pad_sequence(X, batch_first=True)
    mask = torch.arange(padded.size(1))[None, :] < torch.tensor(lengths)[:, None]
    return padded, torch.tensor(y), mask


class GenericTrackDataset(Dataset):
    def __init__(self, X, y):
        self.X = [torch.tensor(x, dtype=torch.float32) for x in X]
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_dataset(path):
    if path.endswith('.pt'):
        data = torch.load(path)
    elif path.endswith('.npy'):
        data = np.load(path, allow_pickle=True).item()
    elif path.endswith('.pkl'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
    else:
        raise ValueError("Unsupported dataset format. Use .pt, .npy, or .pkl")

    if 'X' not in data or 'y' not in data:
        raise ValueError("Dataset must contain 'X' and 'y' keys")
    return data['X'], data['y']


# ----------------- Training Function -----------------
def train(config):
    training_cfg = config['training']
    model_cfg = config['model']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, y = load_dataset(training_cfg['path'])
    dataset = GenericTrackDataset(X, y)

    # Split dataset
    val_len = int(training_cfg['val_split'] * len(dataset))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=training_cfg['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=training_cfg['batch_size'], collate_fn=collate_fn)

    # Dynamically set input dim
    model_cfg['n_input_dims'] = X[0].shape[-1]
    model = TimeSeriesTransformerWithHF(**model_cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg['lr'])

    writer = SummaryWriter(log_dir=training_cfg['log_dir'])
    best_val_loss = float('inf')

    os.makedirs(training_cfg['model_dir'], exist_ok=True)

    for epoch in range(1, training_cfg['epochs'] + 1):
        model.train()
        total_loss = 0
        for Xb, yb, mask in train_loader:
            Xb, yb, mask = Xb.to(device), yb.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(Xb, mask)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # Validation
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for Xb, yb, mask in val_loader:
                Xb, yb, mask = Xb.to(device), yb.to(device), mask.to(device)
                output = model(Xb, mask)
                loss = criterion(output, yb)
                total_val_loss += loss.item()
                preds = output.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = correct / total
        writer.add_scalar("Loss/val", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            model.save_pretrained(training_cfg['model_dir'])
            best_val_loss = avg_val_loss
            print(f"Saved new best model at epoch {epoch}")

    writer.close()

# ----------------- Main Entrypoint -----------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train.py config.yaml")
        exit()

    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    train(config)
