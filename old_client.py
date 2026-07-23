import os
import csv
import argparse
import requests
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import AncestryNet, SUPERPOPS

parser = argparse.ArgumentParser(description="GA4GH DRS Native Stateless Client")
parser.add_argument("--site-id", type=str, required=True, help="Site identifier (a, b, c, or d)")
parser.add_argument(
    "--drs-endpoint",
    type=str,
    required=True,
    help="Base URL of the DRS server (e.g. http://172.17.0.1:4502)",
)
parser.add_argument("--global-weights-path", type=str, required=True, help="Input global checkpoint")
parser.add_argument("--output-weights-path", type=str, required=True, help="Target path for localized weights")
parser.add_argument("--results-dir", type=str, default="./results", help="Metrics storage directory")
parser.add_argument("--epochs", type=int, default=5, help="Local training epochs")
parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
args = parser.parse_args()

SITE_NAME = f"site_{args.site_id}"
RESULTS_DIR = os.path.abspath(args.results_dir)
os.makedirs(RESULTS_DIR, exist_ok=True)

CLIENT_METRICS_CSV = os.path.join(RESULTS_DIR, f"fl_client_{SITE_NAME}_metrics.csv")
FIELDNAMES = ["round", "train_loss", "val_loss", "val_accuracy", "train_samples", "val_samples"]

if not os.path.exists(CLIENT_METRICS_CSV):
    with open(CLIENT_METRICS_CSV, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

def resolve_drs_stream(drs_endpoint, object_id):
    """Resolves a byte URL using a specific site's DRS container."""
    meta_url = f"{drs_endpoint}/ga4gh/drs/v1/objects/{object_id}"
    try:
        meta_resp = requests.get(meta_url, timeout=5).json()
        access_id = meta_resp["access_methods"][0]["access_id"]
        access_url = f"{drs_endpoint}/ga4gh/drs/v1/objects/{object_id}/access/{access_id}"
        stream_url = requests.get(access_url, timeout=5).json()["url"]
        return stream_url.replace("file://", "", 1) if stream_url.startswith("file://") else stream_url
    except Exception:
        raise RuntimeError(
            f"Failed to resolve DRS object '{object_id}' from {drs_endpoint}"
        )


def load_dataset(drs_host, object_id):
    """Ingests dataset via DRS and converts to TensorDataset."""
    stream_url = resolve_drs_stream(drs_host, object_id)
    df = pd.read_csv(stream_url, sep="\t")
    
    pc_cols = sorted([c for c in df.columns if c.upper().startswith("PC")], key=lambda x: int(''.join(filter(str.isdigit, x))))[:10]
    X_tensor = torch.tensor(df[pc_cols].values, dtype=torch.float32)
    y_tensor = torch.tensor(df["super_pop"].apply(lambda x: SUPERPOPS.index(x)).values, dtype=torch.long)
    return TensorDataset(X_tensor, y_tensor)

def main():
    drs_endpoint = args.drs_endpoint
    
    print(f"[{SITE_NAME}] Loading explicit Train/Val splits via {drs_host}...")

    train_dataset = load_dataset(drs_host, f"{SITE_NAME}_train")
    val_dataset = load_dataset(drs_host, f"{SITE_NAME}_val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    net = AncestryNet(input_dim=10, num_classes=5)
    global_state = torch.load(args.global_weights_path, map_location="cpu")
    net.load_state_dict({k: v for k, v in global_state.items() if k != "metadata"}, strict=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    current_round = int(''.join(filter(str.isdigit, os.path.basename(args.output_weights_path).split("round_")[-1])))

    # --- Training Loop ---
    net.train()
    train_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(net(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X_batch.size(0)
        train_loss += epoch_loss
    avg_train_loss = train_loss / (args.epochs * len(train_dataset))

    # --- Validation Loop ---
    net.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = net(X_batch)
            val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)
            correct += (torch.max(outputs, 1)[1] == y_batch).sum().item()

    avg_val_loss = val_loss / len(val_dataset)
    val_accuracy = correct / len(val_dataset)

    print(f"[{SITE_NAME} Round {current_round}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2%}")
    
    with open(CLIENT_METRICS_CSV, mode="a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow({
            "round": current_round, "train_loss": avg_train_loss, "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy, "train_samples": len(train_dataset), "val_samples": len(val_dataset)
        })

    output_payload = net.state_dict()
    output_payload["metadata"] = {"num_examples": len(train_dataset)}
    torch.save(output_payload, args.output_weights_path)

if __name__ == "__main__":
    main()