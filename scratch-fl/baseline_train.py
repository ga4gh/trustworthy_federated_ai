# baseline_train.py
# Centralized baseline: pools all sites' data via the same GA4GH DRS two-step
# resolution used in client.py. Logs one metrics row per FL-equivalent round
# (every EPOCHS_PER_ROUND epochs) so convergence curves are directly comparable.

import os
import csv
import argparse
import requests
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split
from model import AncestryNet, SUPERPOPS

parser = argparse.ArgumentParser(description="Centralized Baseline Trainer")
parser.add_argument("--num-clients",      type=int,   default=4)
parser.add_argument("--rounds",           type=int,   default=10)
parser.add_argument("--epochs-per-round", type=int,   default=5)
parser.add_argument("--batch-size",       type=int,   default=16)
parser.add_argument("--lr",               type=float, default=0.01)
parser.add_argument("--val-fraction",     type=float, default=0.2)
parser.add_argument("--artifacts-dir",    type=str,   default="./checkpoints")
parser.add_argument("--metrics-path",     type=str,   default="./checkpoints/baseline_metrics.csv")
args = parser.parse_args()

os.makedirs(args.artifacts_dir, exist_ok=True)
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASSWISE_CSV    = os.path.join(RESULTS_DIR, "baseline_classwise_metrics.csv")
CLASSWISE_FIELDS = ["round", "loss", "accuracy", "num_examples"] + \
                   [f"acc_{p}" for p in SUPERPOPS] + [f"n_{p}" for p in SUPERPOPS]

# ── GA4GH DRS two-step resolution (mirrors client.py exactly) ─────────────────
def resolve_single_drs_stream(object_id: str, client_id: int) -> str:
    """
    Performs full two-step GA4GH DRS lookup to resolve a streamable file path.
    Sites 1+2 → port 4500, Sites 3+4 → port 4502  (matches client.py routing).
    Falls back to ./data_site_<client_id>_unified.tsv if the Starter Kit is down.
    """
    port    = 4502 if object_id in ("site_3_unified", "site_4_unified") else 4500
    base    = f"http://localhost:{port}/ga4gh/drs/v1/objects/{object_id}"

    try:
        meta_resp = requests.get(base, timeout=5).json()
        access_id = meta_resp["access_methods"][0]["access_id"]

        access_url  = f"{base}/access/{access_id}"
        stream_url  = requests.get(access_url, timeout=5).json()["url"]

        if stream_url.startswith("file://"):
            stream_url = stream_url.replace("file://", "", 1)
        return stream_url

    except Exception as e:
        fallback = f"./data_site_{client_id}_unified.tsv"
        print(f"[DRS Error] {e}  →  falling back to {fallback}")
        return fallback


def load_site_dataset(client_id: int) -> TensorDataset:
    """Resolves DRS and loads the unified TSV for one site into a TensorDataset."""
    unified_id = f"site_{client_id}_unified"
    print(f"\n[Baseline Data] Two-step GA4GH DRS handshake for: {unified_id}")

    path = resolve_single_drs_stream(unified_id, client_id)
    print(f"[Baseline Data] Stream resolved → {path}")

    df      = pd.read_csv(path, sep="\t")
    pc_cols = sorted(
        [c for c in df.columns if c.upper().startswith("PC") and c.upper().endswith("_AVG")],
        key=lambda c: int("".join(filter(str.isdigit, c)))
    )[:10]

    X = torch.tensor(df[pc_cols].values, dtype=torch.float32)
    y = torch.tensor(df["super_pop"].apply(lambda x: SUPERPOPS.index(x)).values, dtype=torch.long)
    return TensorDataset(X, y)


# ── Helpers ───────────────────────────────────────────────────────────────────
def classwise_accuracy(y_true, y_pred):
    accs, counts = {}, {}
    for idx, pop in enumerate(SUPERPOPS):
        mask        = (y_true == idx)
        n           = mask.sum().item()
        counts[pop] = n
        accs[pop]   = (((y_pred == y_true) & mask).sum().item() / n) if n > 0 else 0.0
    return accs, counts


def evaluate(net, loader, criterion):
    net.eval()
    total_loss, correct = 0.0, 0
    all_preds, all_true = [], []
    with torch.no_grad():
        for X, y in loader:
            out        = net(X)
            total_loss += criterion(out, y).item() * X.size(0)
            preds       = out.argmax(dim=1)
            correct    += (preds == y).sum().item()
            all_preds.append(preds)
            all_true.append(y)
    n             = len(loader.dataset)
    c_accs, c_cnts = classwise_accuracy(torch.cat(all_true), torch.cat(all_preds))
    return total_loss / n, correct / n, c_accs, c_cnts


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("   Centralized Baseline Training  (FL-equivalent round checkpoints)")
    print("=" * 70)

    # Resolve and pool all site datasets via DRS
    all_datasets = [load_site_dataset(i) for i in range(1, args.num_clients + 1)]
    full_dataset  = ConcatDataset(all_datasets)

    total_samples = len(full_dataset)
    n_val         = max(1, int(total_samples * args.val_fraction))
    n_train       = total_samples - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    print(f"\n[Baseline] Pooled dataset: {total_samples} total  "
          f"({n_train} train | {n_val} val)  across {args.num_clients} sites")

    net       = AncestryNet(input_dim=10, num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Write CSV headers
    with open(args.metrics_path, mode="w", newline="") as f:
        csv.DictWriter(f, fieldnames=["round", "weighted_loss", "weighted_accuracy", "total_samples"]).writeheader()
    with open(CLASSWISE_CSV, mode="w", newline="") as f:
        csv.DictWriter(f, fieldnames=CLASSWISE_FIELDS).writeheader()

    for r in range(1, args.rounds + 1):
        # Train for one FL-equivalent round
        net.train()
        for _ in range(args.epochs_per_round):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(net(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        val_loss, val_acc, c_accs, c_cnts = evaluate(net, val_loader, criterion)
        print(f"[Baseline Round {r:>2}/{args.rounds}]  "
              f"Loss: {val_loss:.4f}  |  Accuracy: {val_acc*100:.2f}%")

        with open(args.metrics_path, mode="a", newline="") as f:
            csv.DictWriter(f, fieldnames=["round", "weighted_loss", "weighted_accuracy", "total_samples"]).writerow({
                "round": r, "weighted_loss": val_loss,
                "weighted_accuracy": val_acc, "total_samples": total_samples,
            })

        with open(CLASSWISE_CSV, mode="a", newline="") as f:
            row = {"round": r, "loss": val_loss, "accuracy": val_acc, "num_examples": total_samples}
            row.update({f"acc_{p}": c_accs[p] for p in SUPERPOPS})
            row.update({f"n_{p}":   c_cnts[p]  for p in SUPERPOPS})
            csv.DictWriter(f, fieldnames=CLASSWISE_FIELDS).writerow(row)

    final_path = os.path.join(args.artifacts_dir, "baseline_model_final.pt")
    torch.save(net.state_dict(), final_path)
    print(f"\n[Baseline] Final model → {final_path}")
    print(f"[Baseline] Metrics     → {args.metrics_path}")
    print(f"[Baseline] Classwise   → {CLASSWISE_CSV}")


if __name__ == "__main__":
    main()