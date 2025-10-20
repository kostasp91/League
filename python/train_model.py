#!/usr/bin/env python3
"""
Rewritten train_model.py

Upgrades:
- Robust feature selection (auto-includes roll3FPT if present)
- Early stopping on validation loss
- Reproducibility via seeds
- Saves features_used.json and metrics.json alongside model & scaler
- Clear logs every N epochs

Usage:
  python train_model.py \
      --train_csv ../TrainingModel/processed_train_dataset.csv \
      --out_dir ../TrainingModel \
      --epochs 300 \
      --patience 30 \
      --batch_size 256
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PlayerModel(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def log(msg: str):
    print(f"[train_model] {msg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="../TrainingModel/processed_train_dataset.csv")
    parser.add_argument("--out_dir", type=str, default="../TrainingModel")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(args.train_csv)
    df = df.copy()

    # Base features (keep parity with server expectations)
    base_features = [
        "CR", "MIN", "avgFPT", "lastFPT", "adjEfficiency",
        "avg_REB", "avg_AST", "avg_STL", "avg_BLK", "avg_TOV", "avg_PF",
        "FG_pct", "TP_pct", "FT_pct",
    ]
    # Optional feature (if present)
    optional_features = ["roll3FPT"]

    features = [f for f in base_features if f in df.columns]
    features += [f for f in optional_features if f in df.columns]

    target = "NextFantasyPoints"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {args.train_csv}")

    # Filter rows with target present
    df = df[df[target].notna()].reset_index(drop=True)

    # Fill NA for features
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    X = df[features].values
    y = df[target].values.astype(np.float32)

    # Scale + Split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=args.val_size, random_state=args.seed
    )

    # Torch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

    # Model
    model = PlayerModel(input_size=X_train_t.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Early stopping setup
    best_val = float("inf")
    best_state = None
    patience = args.patience
    bad_epochs = 0

    def run_epoch(split="train"):
        if split == "train":
            model.train()
            # Simple full-batch or mini-batch training
            bs = args.batch_size
            indices = np.arange(len(X_train_t))
            np.random.shuffle(indices)
            total_loss = 0.0
            for start in range(0, len(indices), bs):
                end = start + bs
                batch_idx = indices[start:end]
                xb = X_train_t[batch_idx]
                yb = y_train_t[batch_idx]

                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_idx)
            return total_loss / len(indices)
        else:
            model.eval()
            with torch.no_grad():
                pred = model(X_val_t)
                loss = criterion(pred, y_val_t).item()
            return loss

    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch("train")
        val_loss = run_epoch("val")

        if epoch == 1 or epoch % 20 == 0:
            log(f"Epoch {epoch:3d}/{args.epochs} | Train MSE: {tr_loss:.4f} | Val MSE: {val_loss:.4f}")

        # early stopping check
        if (val_loss < best_val - 1e-6):
            best_val = val_loss;
            best_state = model.state_dict()
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                log(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # Finalize model
    if (best_state is not None):
        model.load_state_dict(best_state)

    # Evaluate final metrics on val
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_t).squeeze(1).cpu().numpy()
    metrics = {
        "val_mae": mae(y_val, y_val_pred),
        "val_mse": mse(y_val, y_val_pred),
        "val_rmse": float(np.sqrt(mse(y_val, y_val_pred))),
        "best_val_mse": float(best_val),
        "num_features": len(features),
        "features": features,
    }

    # Save artifacts
    model_path = os.path.join(args.out_dir, "player_model.pth")
    scaler_path = os.path.join(args.out_dir, "scaler.pkl")
    features_path = os.path.join(args.out_dir, "features_used.json")
    metrics_path = os.path.join(args.out_dir, "metrics.json")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    with open(features_path, "w", encoding="utf-8") as f:
        json.dump({"features": features}, f, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log("Model trained and saved.")
    log(f"Saved: {model_path}")
    log(f"Saved: {scaler_path}")
    log(f"Saved: {features_path}")
    log(f"Saved: {metrics_path}")
    log(f"Features used: {features}")
    log(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    main()
