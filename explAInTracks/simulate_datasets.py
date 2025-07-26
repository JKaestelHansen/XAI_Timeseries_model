# datasets.py
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import os
import argparse
import pickle


def simulate_brownian(N, D=1.0, dt=1.0):
    steps = np.random.normal(scale=np.sqrt(2 * D * dt), size=(N, 3))
    return np.cumsum(steps, axis=0)


def simulate_directed(N, v=[1.0, 0.0, 0.0], D=0.2, dt=1.0):
    drift = np.outer(np.arange(N), v)
    noise = np.random.normal(scale=np.sqrt(2 * D * dt), size=(N, 3))
    return drift + np.cumsum(noise, axis=0)


def simulate_confined(N, radius=5.0, D=1.0, dt=1.0):
    x = np.zeros((N, 3))
    for i in range(1, N):
        step = np.random.normal(scale=np.sqrt(2 * D * dt), size=3)
        x[i] = x[i - 1] + step
        if np.linalg.norm(x[i]) > radius:
            x[i] = x[i - 1] - step  # reflect back
    return x


def simulate_anomalous(N, hurst=0.3, scale=1.0):
    from fbm import FBM
    tracks = []
    for _ in range(3):
        f = FBM(n=N - 1, hurst=hurst, length=1.0, method="daviesharte")
        tracks.append(scale * f.fbm())
    return np.stack(tracks, axis=1)


MOTION_TYPES = ['brownian', 'directed', 'confined', 'anomalous']
def generate_track(N, motion_type):
    if motion_type == 'brownian':
        return simulate_brownian(N)
    elif motion_type == 'directed':
        return simulate_directed(N)
    elif motion_type == 'confined':
        return simulate_confined(N)
    elif motion_type == 'anomalous':
        return simulate_anomalous(N)
    else:
        raise ValueError(f"Unknown motion type: {motion_type}")

class MotionDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=100, motion_types=MOTION_TYPES):
        self.X = []
        self.y = []
        for _ in range(num_samples):
            motion = random.choice(motion_types)
            x = generate_track(seq_len, motion)
            self.X.append(torch.tensor(x, dtype=torch.float32))
            self.y.append(motion_types.index(motion))
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def generate_dataset(num_samples=1000, seq_len=100, output_path="data/synthetic_dataset.pt"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    X = []
    y = []

    for i in range(num_samples):
        motion = np.random.choice(MOTION_TYPES)
        track = generate_track(seq_len, motion)
        X.append(track.astype(np.float32))
        y.append(MOTION_TYPES.index(motion))

    # Save as PyTorch dictionary
    data = {"X": X, "y": y}
    if output_path.endswith(".pt"):
        torch.save(data, output_path)
    elif output_path.endswith(".pkl"):
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
    elif output_path.endswith(".npy"):
        np.save(output_path, data)
    else:
        raise ValueError("Unsupported format. Use .pt, .pkl, or .npy")

    print(f"Saved {num_samples} tracks to: {output_path}")

if __name__ == "__main__":
    import sys
    import yaml

    if len(sys.argv) != 2:
        print("Usage: python train.py config.yaml")
        exit()

    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    try:
        num_samples = config['simulation']['num_samples']
        seq_len = config['simulation']['seq_len']
        output_path = config['simulation']['output_path']
    except KeyError:
        print("did not contain expected keys, using defaults")
        num_samples = 1000
        seq_len = 100
        output_path = "data/synthetic_dataset.pt"

    generate_dataset(num_samples, seq_len, output_path)