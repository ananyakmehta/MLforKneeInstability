#test model jk only

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# =====================================================
# MULTI-TRIAL SETTINGS
# =====================================================
NUM_TRIALS = 50
VERBOSE_TRAINING = False


# -----------------------------
# Config
# -----------------------------
DATA_DIR = "generated"
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# IMPORTANT: Set this correctly
# -----------------------------
NUM_IK_CHANNELS = 30   # <-- change this to match your dataset


# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")

X = np.load(os.path.join(DATA_DIR, "X.npy"))  # (samples, 72, 39)
y = np.load(os.path.join(DATA_DIR, "y.npy"))

# transpose to (N, channels, timesteps)
X = np.transpose(X, (0, 2, 1))  # (N, 39, 72)

# -----------------------------
# Keep ONLY IK channels
# -----------------------------
X = X[:, :NUM_IK_CHANNELS, :]   # (N, IK_channels, 72)

print(f"Using IK-only input shape: {X.shape}")


# -----------------------------
# CNN Model
# -----------------------------
class CNN1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, 5)
        self.conv2 = nn.Conv1d(64, 128, 5)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 4)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


def train_model(X_train, y_train, X_val, y_val, in_channels):
    model = CNN1D(in_channels).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        permutation = torch.randperm(X_train.size(0))

        for i in range(0, X_train.size(0), BATCH_SIZE):
            idx = permutation[i:i+BATCH_SIZE]
            batch_x = X_train[idx]
            batch_y = y_train[idx]

            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        if VERBOSE_TRAINING:
            model.eval()
            with torch.no_grad():
                preds = torch.argmax(model(X_val), dim=1)
                acc = accuracy_score(y_val.cpu(), preds.cpu())
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Val Acc: {acc*100:.2f}%")

    return model


# =====================================================
# 50-TRIAL EVALUATION
# =====================================================

print("\nTRIAL,TestAccuracy(%)")

all_acc = []

for trial in range(NUM_TRIALS):

    torch.manual_seed(trial)
    np.random.seed(trial)

    # --- split ---
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, stratify=y_temp
    )

    # --- scaling ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(len(X_train), -1))
    X_val = scaler.transform(X_val.reshape(len(X_val), -1))
    X_test = scaler.transform(X_test.reshape(len(X_test), -1))

    # reshape back to CNN format
    X_train = torch.tensor(
        X_train.reshape(-1, NUM_IK_CHANNELS, 72),
        dtype=torch.float32
    ).to(DEVICE)

    X_val = torch.tensor(
        X_val.reshape(-1, NUM_IK_CHANNELS, 72),
        dtype=torch.float32
    ).to(DEVICE)

    X_test = torch.tensor(
        X_test.reshape(-1, NUM_IK_CHANNELS, 72),
        dtype=torch.float32
    ).to(DEVICE)

    y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    y_val = torch.tensor(y_val, dtype=torch.long).to(DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

    # --- train ---
    model = train_model(
        X_train, y_train, X_val, y_val,
        in_channels=NUM_IK_CHANNELS
    )

    # --- test ---
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1)
        test_acc = accuracy_score(y_test.cpu(), preds.cpu())

    all_acc.append(test_acc)

    print(f"{trial+1},{test_acc*100:.2f}")


# =====================================================
# SUMMARY
# =====================================================
print("\nSUMMARY")
print(f"MeanAccuracy,{np.mean(all_acc)*100:.2f}")
print(f"StdAccuracy,{np.std(all_acc)*100:.2f}")