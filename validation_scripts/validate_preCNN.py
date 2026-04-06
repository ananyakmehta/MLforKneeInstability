import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = "generated"

# -----------------------------
# Load dataset
# -----------------------------
def load_dataset():

    X = np.load(os.path.join(DATA_DIR, "X.npy"))   # (samples, 72, 39)
    y = np.load(os.path.join(DATA_DIR, "y.npy"))

    print(f"Raw shape: {X.shape}")

    # Flatten for classical ML
    X_flat = X.reshape(X.shape[0], -1)

    return X, X_flat, y


# -----------------------------
# Severity Scaling Check
# -----------------------------
def severity_scaling_check(X, y):
    print("\n--- Severity Scaling Check ---")

    for s in sorted(np.unique(y)):
        idx = y == s
        mean_abs = np.mean(np.abs(X[idx]))
        print(f"S{s}: Mean Abs Torque = {mean_abs:.2f}")


# -----------------------------
# PCA Visualization
# -----------------------------
def pca_visualization(X_flat, y):

    print("\n--- PCA Separability Check ---")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_scaled)

    plt.figure()
    for s in np.unique(y):
        idx = y == s
        plt.scatter(X_reduced[idx, 0], X_reduced[idx, 1], label=f"S{s}", alpha=0.6)

    plt.legend()
    plt.title("PCA of ID Torque Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


# -----------------------------
# Logistic Regression Baseline
# -----------------------------
def logistic_baseline(X_flat, y):

    print("\n--- Logistic Regression Baseline ---")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Random Split Accuracy: {acc*100:.2f}%")

    return acc


# -----------------------------
# Binary Baseline (Optional Diagnostic)
# Collapse S0+S1 vs S2+S3
# -----------------------------
def binary_diagnostic(X_flat, y):

    print("\n--- Binary Low vs High Severity Diagnostic ---")

    y_binary = (y >= 2).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_binary,
        test_size=0.2,
        random_state=42,
        stratify=y_binary
    )

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Binary Accuracy: {acc*100:.2f}%")

    return acc


# -----------------------------
# Main
# -----------------------------
def main():

    print("Loading dataset...")
    X, X_flat, y = load_dataset()

    print(f"Samples: {X.shape[0]}")
    print(f"Timesteps: {X.shape[1]}")
    print(f"Features per timestep: {X.shape[2]}")
    print(f"Flattened features: {X_flat.shape[1]}")

    severity_scaling_check(X, y)
    pca_visualization(X_flat, y)

    acc_multi = logistic_baseline(X_flat, y)
    acc_binary = binary_diagnostic(X_flat, y)

    print("\n--- Interpretation ---")

    if acc_multi > 0.6:
        print("Strong multiclass signal detected.")
    elif acc_multi > 0.45:
        print("Moderate signal detected. CNN likely beneficial.")
    else:
        print("Weak linear signal. Consider strengthening perturbations.")

    if acc_binary > acc_multi:
        print("Binary separation is stronger than fine-grained severity separation.")


if __name__ == "__main__":
    main()