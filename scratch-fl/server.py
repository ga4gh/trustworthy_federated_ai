import os
import csv
import logging
import argparse
import requests
import pandas as pd
import torch
import torch.nn as nn
from io import StringIO
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

def resolve_central_test_data(drs_endpoint, object_id="central_test"):
    """Fetches global test set strictly from central DRS node."""
    try:
        drs_endpoint = drs_endpoint.rstrip("/")
        meta_url = f"{drs_endpoint}/ga4gh/drs/v1/objects/{object_id}"
        
        meta_resp = requests.get(meta_url, timeout=5)
        meta_resp.raise_for_status()
        
        access_id = meta_resp.json()["access_methods"][0]["access_id"]
        access_url = f"{drs_endpoint}/ga4gh/drs/v1/objects/{object_id}/access/{access_id}"
        access_resp = requests.get(access_url, timeout=5)
        access_resp.raise_for_status()
        
        stream_url = access_resp.json()["url"]
        
        if "localhost:4500" in stream_url:
            stream_url = stream_url.replace("http://localhost:4500", drs_endpoint, 1)
            
        stream_url = stream_url.replace("\n", "").replace("file://", "", 1)
        logger.info(f"Resolved test data stream URL: {stream_url}")
        
        stream_resp = requests.get(stream_url, timeout=10)
        stream_resp.raise_for_status()

        if not stream_resp.text.strip():
            raise ValueError(f"DRS stream at '{stream_url}' returned 0 bytes / empty content.")

        df = pd.read_csv(StringIO(stream_resp.text), sep="\t")
        pc_cols = sorted([c for c in df.columns if c.upper().startswith("PC")], key=lambda x: int(''.join(filter(str.isdigit, x))))[:10]
        
        X = torch.tensor(df[pc_cols].values, dtype=torch.float32)
        y = torch.tensor(df["super_pop"].apply(lambda x: SUPERPOPS.index(x)).values, dtype=torch.long)
        return DataLoader(TensorDataset(X, y), batch_size=32, shuffle=False)
        
    except Exception as e:
        logger.error(f"Failed to fetch test data from DRS node '{drs_endpoint}' for object '{object_id}': {e}")
        raise RuntimeError(f"Strict evaluation failure: Central DRS test data could not be resolved.") from e

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
    parser.add_argument("--drs-endpoint", type=str, default="http://172.17.0.1:4500", help="Base URL of Central DRS server")
    parser.add_argument("--sites", type=str, default="1,2,3,4", help="Comma-separated site IDs (e.g. 1,2,3,4 or site_1,site_2)")
    parser.add_argument("--artifacts-dir", type=str, required=True, default="./checkpoints")
    parser.add_argument("--metrics-path", type=str, required=True, help="Explicit per-round metrics file output path")
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.metrics_path)), exist_ok=True)
    global_model = AncestryNet(input_dim=10, num_classes=5)
    
    # ✅ Fixed: Correct variable parsing
    sites = [s.strip().replace("site_", "") for s in args.sites.split(",") if s.strip()]

    # Bootstrap Round 0
    if args.target_round == 0:
        torch.save(global_model.state_dict(), os.path.join(args.artifacts_dir, "global_model_round_0.pt"))
        with open(args.metrics_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "global_test_loss", "global_test_accuracy", "total_train_samples"])
            writer.writeheader()
            writer.writerow({
                "round": 0,
                "global_test_loss": 0.0,
                "global_test_accuracy": 0.0,
                "total_train_samples": 0
            })
        return

    r = args.target_round
    logger.info(f"--- Federated Weight Aggregation for Round {r} ---")
    
    client_checkpoints, client_sample_sizes = [], []
    for site in sites:
        path = os.path.join(args.artifacts_dir, f"client_{site}_round_{r}.pt")
        if not os.path.exists(path):
            alt_path = os.path.join(args.artifacts_dir, f"client_site_{site}_round_{r}.pt")
            if os.path.exists(alt_path):
                path = alt_path

        payload = torch.load(path, map_location="cpu")
        client_checkpoints.append(path)
        client_sample_sizes.append(payload.get("metadata", {}).get("num_examples", 100))

    # Aggregation
    new_state_dict = federated_averaging(global_model, client_checkpoints, client_sample_sizes)
    torch.save(new_state_dict, os.path.join(args.artifacts_dir, f"global_model_round_{r}.pt"))
    
    # Load weights into model and evaluate on Central Test Set via dynamic DRS endpoint
    global_model.load_state_dict(new_state_dict, strict=False)
    test_loader = resolve_central_test_data(args.drs_endpoint)
    test_loss, test_acc = evaluate_global_model(global_model, test_loader)
    total_train_samples = sum(client_sample_sizes)

    logger.info(f"[Round {r} Complete] Global Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")

    # Write per-round metrics file directly
    fieldnames = ["round", "global_test_loss", "global_test_accuracy", "total_train_samples"]
    with open(args.metrics_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "round": r,
            "global_test_loss": test_loss,
            "global_test_accuracy": test_acc,
            "total_train_samples": total_train_samples
        })

if __name__ == "__main__":
    main()