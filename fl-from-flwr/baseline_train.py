# baseline_train.py
"""
Centralized baseline: trains AncestryNet on the FULL pooled dataset
(no federation, no client/server split). This is the reference
convergence curve you compare every FL run against.

Usage:
    python baseline_train.py

Expects the real FLAN sscore cache, same path used by DRS/extract_from_flan.py:
    /home/viditkh/.cache/deep_ancestry/genotypes/fold_0/train_genotype.sscore

Outputs:
    results/baseline_metrics.csv   (epoch, train_loss, train_acc, val_loss, val_acc)
    results/baseline_curve.png     (loss + accuracy convergence plot)
"""
import os
import sys
import csv

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

# reuse the exact model + label ordering used by the FL clients, no drift allowed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "fl-from-scratch"))
from model import AncestryNet, SUPERPOPS  # noqa: E402

SSCORE_PATH = "/home/viditkh/.cache/deep_ancestry/genotypes/fold_0/train_genotype.sscore"
METADATA_URL = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20130606_sample_info/20130606_sample_info.txt"

POPULATION_TO_SUPERPOP = {
    'CEU': 'EUR', 'TSI': 'EUR', 'FIN': 'EUR', 'GBR': 'EUR', 'IBS': 'EUR',
    'YRI': 'AFR', 'LWK': 'AFR', 'GWD': 'AFR', 'MSL': 'AFR', 'ESN': 'AFR', 'ASW': 'AFR', 'ACB': 'AFR',
    'CHB': 'EAS', 'JPT': 'EAS', 'CHS': 'EAS', 'CDX': 'EAS', 'KHV': 'EAS',
    'GIH': 'SAS', 'PJL': 'SAS', 'BEB': 'SAS', 'STU': 'SAS', 'ITU': 'SAS',
    'MXL': 'AMR', 'PUR': 'AMR', 'CLM': 'AMR', 'PEL': 'AMR',
}

NUM_EPOCHS = 15
BATCH_SIZE = 32
LR = 0.005
VAL_FRACTION = 0.2
SEED = 0
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_pooled_dataset():
    print(f"Reading real preprocessed features from: {SSCORE_PATH}")
    df = pd.read_csv(SSCORE_PATH, sep=r'\s+')
    df.rename(columns={'#IID': 'IID', '#FID': 'FID'}, inplace=True)

    print("Downloading population reference panel metadata maps...")
    meta_df = pd.read_csv(METADATA_URL, sep='\t')
    pop_map = dict(zip(meta_df['Sample'], meta_df['Population']))

    super_pops = [POPULATION_TO_SUPERPOP.get(pop_map.get(iid), 'UNKNOWN') for iid in df['IID']]
    df['super_pop'] = super_pops
    df = df[df['super_pop'] != 'UNKNOWN'].reset_index(drop=True)

    pc_cols = [c for c in df.columns if c.upper().startswith("PC") and c.upper().endswith("_AVG")]
    pc_cols = sorted(pc_cols, key=lambda c: int(''.join(filter(str.isdigit, c))))
    print(f"Using {len(pc_cols)} PCA features: {pc_cols[:3]}...{pc_cols[-1]}")

    X = df[pc_cols].values.astype(np.float32)
    y = df['super_pop'].apply(lambda s: SUPERPOPS.index(s)).values.astype(np.int64)

    print(f"Pooled dataset size: {len(df)} individuals across {len(SUPERPOPS)} superpopulations")
    print("Class counts:", {SUPERPOPS[c]: int((y == c).sum()) for c in range(len(SUPERPOPS))})

    return TensorDataset(torch.tensor(X), torch.tensor(y)), len(pc_cols)


def epoch_pass(net, loader, criterion, optimizer=None):
    """One pass over loader. If optimizer is given, trains; else evaluates."""
    is_train = optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for X_batch, y_batch in loader:
            if is_train:
                optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            correct += (torch.argmax(outputs, 1) == y_batch).sum().item()
            total += X_batch.size(0)
    return total_loss / total, correct / total


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    dataset, input_dim = load_pooled_dataset()

    n_val = int(len(dataset) * VAL_FRACTION)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(SEED)
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    net = AncestryNet(input_dim=input_dim, num_classes=len(SUPERPOPS))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    rows = []
    print(f"\nTraining centralized baseline for {NUM_EPOCHS} epochs "
          f"({n_train} train / {n_val} val individuals)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = epoch_pass(net, train_loader, criterion, optimizer)
        val_loss, val_acc = epoch_pass(net, val_loader, criterion, optimizer=None)
        rows.append([epoch, train_loss, train_acc, val_loss, val_acc])
        if epoch == 1 or epoch % 5 == 0 or epoch == NUM_EPOCHS:
            print(f"  epoch {epoch:3d} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                  f"| val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    metrics_path = os.path.join(RESULTS_DIR, "baseline_metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writerows(rows)
    print(f"\nSaved per-epoch metrics -> {metrics_path}")

    epochs, tr_loss, tr_acc, va_loss, va_acc = zip(*rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    axes[0].plot(epochs, tr_loss, label="train")
    axes[0].plot(epochs, va_loss, label="val")
    axes[0].set_title("Baseline Loss (centralized)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].legend()

    axes[1].plot(epochs, tr_acc, label="train")
    axes[1].plot(epochs, va_acc, label="val")
    axes[1].set_title("Baseline Accuracy (centralized)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend()

    fig.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, "baseline_curve.png")
    fig.savefig(curve_path, dpi=150)
    print(f"Saved convergence plot -> {curve_path}")
    print(f"\nFinal val accuracy: {va_acc[-1]*100:.2f}% | Final val loss: {va_loss[-1]:.4f}")


if __name__ == "__main__":
    main()
