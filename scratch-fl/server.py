# server.py
import os
import csv
import logging
import argparse
from typing import List, Dict, Any
import torch
from model import AncestryNet

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][Orchestrator Server] %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def federated_averaging(global_model: torch.nn.Module, client_checkpoints: List[str], client_sample_sizes: List[int]) -> Dict[str, Any]:
    """Natively applies the FedAvg algorithm across reporting client parameters."""
    total_samples = sum(client_sample_sizes)
    if total_samples == 0:
        raise ValueError("Total sample count across reporting nodes equals zero.")

    global_state = global_model.state_dict()
    aggregated_state = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in global_state.items()}

    logger.info(f"Aggregating {len(client_checkpoints)} client state dicts across {total_samples} cumulative samples.")

    for path, sample_size in zip(client_checkpoints, client_sample_sizes):
        client_state = torch.load(path, map_location="cpu")
        weight = sample_size / total_samples
        for k in aggregated_state.keys():
            # Avoid aggregating embedded metadata wrappers
            if k == "metadata":
                continue
            aggregated_state[k] += client_state[k].float() * weight

    return aggregated_state

def main():
    parser = argparse.ArgumentParser(description="Stateless GA4GH-Aligned FedAvg Step Core")
    parser.add_argument("--target-round", type=int, required=True, help="The exact unified training round step to process")
    parser.add_argument("--num-clients", type=int, default=4, help="Expected count of remote data partitions")
    parser.add_argument("--artifacts-dir", type=str, default="./checkpoints", help="Directory for checkpoint storage")
    parser.add_argument("--metrics-path", type=str, default="./checkpoints/server_metrics.csv", help="CSV metrics output path")
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)
    global_model = AncestryNet(input_dim=10, num_classes=5)

    # Base Initial Setup Phase (Round 0 Bootstrap logic called at start of pipeline)
    if args.target_round == 0:
        initial_weights_path = os.path.join(args.artifacts_dir, "global_model_round_0.pt")
        torch.save(global_model.state_dict(), initial_weights_path)
        logger.info(f"Initialized global architecture base weights saved to {initial_weights_path}")
        
        # Fresh metrics file initialization
        fieldnames = ["round", "weighted_loss", "weighted_accuracy", "total_samples"]
        with open(args.metrics_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return

    # --- Processing target discrete weight aggregation step ---
    r = args.target_round
    logger.info(f"--- Processing Federated Weight Aggregation for Round {r}/{r} ---")
    
    client_checkpoints = []
    client_sample_sizes = []
    round_losses = []
    round_accuracies = []

    for c_id in range(1, args.num_clients + 1):
        client_out_path = os.path.join(args.artifacts_dir, f"client_{c_id}_round_{r}.pt")
        
        if not os.path.exists(client_out_path):
            raise FileNotFoundError(f"Checkpoint signature {client_out_path} missing. Awaiting execution runtime context.")

        client_checkpoints.append(client_out_path)
        
        # Load file safely to parse data total allocations
        payload = torch.load(client_out_path, map_location="cpu")
        meta = payload.get("metadata", {})
        
        c_samples = meta.get("num_examples", 100)
        c_loss = meta.get("val_loss", 0.0)
        c_acc = meta.get("val_accuracy", 0.0)
        
        client_sample_sizes.append(c_samples)
        round_losses.append(c_loss * c_samples)
        round_accuracies.append(c_acc * c_samples)

    # Compute Federated Averaging parameters adjustments
    new_state_dict = federated_averaging(global_model, client_checkpoints, client_sample_sizes)
    
    # Save the updated master parameter layer adjustments for next sequence step
    next_global_path = os.path.join(args.artifacts_dir, f"global_model_round_{r}.pt")
    torch.save(new_state_dict, next_global_path)
    
    # Calculate global metrics
    total_round_samples = sum(client_sample_sizes)
    agg_loss = sum(round_losses) / total_round_samples
    agg_accuracy = sum(round_accuracies) / total_round_samples

    logger.info(f"[Round {r} Complete] Aggregated Loss: {agg_loss:.4f} | Accuracy: {agg_accuracy*100:.2f}%")

    # Record metrics line row to CSV tracking sheet
    fieldnames = ["round", "weighted_loss", "weighted_accuracy", "total_samples"]
    with open(args.metrics_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            "round": r,
            "weighted_loss": agg_loss,
            "weighted_accuracy": agg_accuracy,
            "total_samples": total_round_samples
        })

if __name__ == "__main__":
    main()