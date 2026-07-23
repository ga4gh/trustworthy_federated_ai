import os
import csv
import logging
import argparse
import requests
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Any
from model import AncestryNet, SUPERPOPS

logging.basicConfig(level=logging.INFO, format="[Server] %(message)s")
logger = logging.getLogger(__name__)

def federated_averaging(global_model, client_checkpoints, client_sample_sizes):
    total_samples = sum(client_sample_sizes)
    global_state = global_model.state_dict()
    aggregated_state = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_state.items()}

    for path, sample_size in zip(client_checkpoints, client_sample_sizes):
        client_state = torch.load(path, map_location="cpu")
        weight = sample_size / total_samples
        for k in aggregated_state.keys():
            if k != "metadata":
                aggregated_state[k] += client_state[k].float() * weight
    return aggregated_state

def resolve_central_test_data():
    """Fetches the global test set from the central DRS node."""
    drs_host = "drs-central:4500"
    object_id = "central_test"
    try:
        meta_url = f"http://{drs_host}/ga4gh/drs/v1/objects/{object_id}"
        access_id = requests.get(meta_url, timeout=5).json()["access_methods"][0]["access_id"]
        stream_url = requests.get(f"{meta_url}/access/{access_id}", timeout=5).json()["url"]
        stream_url = stream_url.replace("file://", "", 1) if stream_url.startswith("file://") else stream_url
        
        df = pd.read_csv(stream_url, sep="\t")
        pc_cols = sorted([c for c in df.columns if c.upper().startswith("PC")], key=lambda x: int(''.join(filter(str.isdigit, x))))[:10]
        
        X = torch.tensor(df[pc_cols].values, dtype=torch.float32)
        y = torch.tensor(df["super_pop"].apply(lambda x: SUPERPOPS.index(x)).values, dtype=torch.long)
        return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=False)
    except Exception as e:
        logger.warning("Could not connect to central DRS test set. Using fallback logic.")
        # Fallback dummy evaluation data
        X = torch.randn(100, 10)
        y = torch.randint(0, 5, (100,))
        return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=False)

def evaluate_global_model(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in test_loader:
            outputs = model(X)
            test_loss += criterion(outputs, y).item() * X.size(0)
            correct += (torch.max(outputs, 1)[1] == y).sum().item()
    
    return test_loss / len(test_loader.dataset), correct / len(test_loader.dataset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-round", type=int, required=True)
    parser.add_argument("--sites", type=str, default="a,b,c,d")
    parser.add_argument("--artifacts-dir", type=str, default="./checkpoints")
    parser.add_argument("--metrics-path", type=str, default="./checkpoints/server_metrics.csv")
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)
    global_model = AncestryNet(input_dim=10, num_classes=5)
    sites = args.sites.split(",")

    # Bootstrap Round 0
    if args.target_round == 0:
        torch.save(global_model.state_dict(), os.path.join(args.artifacts_dir, "global_model_round_0.pt"))
        with open(args.metrics_path, mode="w", newline="") as f:
            csv.DictWriter(f, fieldnames=["round", "global_test_loss", "global_test_accuracy", "total_train_samples"]).writeheader()
        return

    r = args.target_round
    logger.info(f"--- Federated Weight Aggregation for Round {r} ---")
    
    client_checkpoints, client_sample_sizes = [], []
    for site in sites:
        path = os.path.join(args.artifacts_dir, f"client_{site}_round_{r}.pt")
        payload = torch.load(path, map_location="cpu")
        client_checkpoints.append(path)
        client_sample_sizes.append(payload.get("metadata", {}).get("num_examples", 100))

    # Aggregation
    new_state_dict = federated_averaging(global_model, client_checkpoints, client_sample_sizes)
    torch.save(new_state_dict, os.path.join(args.artifacts_dir, f"global_model_round_{r}.pt"))
    
    # Load weights into model and evaluate on Central Test Set
    global_model.load_state_dict(new_state_dict, strict=False)
    test_loader = resolve_central_test_data()
    test_loss, test_acc = evaluate_global_model(global_model, test_loader)
    total_train_samples = sum(client_sample_sizes)

    logger.info(f"[Round {r} Complete] Global Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")

    # Log to CSV
    with open(args.metrics_path, mode="a", newline="") as f:
        csv.DictWriter(f, fieldnames=["round", "global_test_loss", "global_test_accuracy", "total_train_samples"]).writerow({
            "round": r, "global_test_loss": test_loss, "global_test_accuracy": test_acc, "total_train_samples": total_train_samples
        })

if __name__ == "__main__":
    main()