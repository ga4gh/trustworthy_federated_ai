"""
new_client.py

Ephemeral, single-shot Federated Learning client for genomics ancestry
prediction. Designed to be invoked as a single GA4GH TES task: it does
exactly one unit of work (load global weights -> train locally -> write
updated weights) and terminates. No persistent process, no socket
connection to a server, no Flower (flwr) dependency.

Lifecycle for one invocation:
    1. Resolve the local PC-coordinate dataset (via DRS, or a plain local
       path -- both supported).
    2. Load the global model weights from --global-weights-path.
    3. Run a short local training routine.
    4. Atomically write the updated state dict + sample-count metadata to
       --output-weights-path / its sibling metadata file.
    5. Write a ".done" signal file the orchestrator polls on.
    6. Exit.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import AncestryNet, SUPERPOPS

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Client] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("client")


# --------------------------------------------------------------------------- #
# Atomic file I/O (same contract as the server side)
# --------------------------------------------------------------------------- #

def atomic_torch_save(obj, dest_path: Path) -> None:
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dest_path.parent), suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_json_save(obj, dest_path: Path) -> None:
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dest_path.parent), suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        os.replace(tmp_path, dest_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def wait_for_file(path: Path, timeout_s: float, poll_interval_s: float = 0.5) -> bool:
    """Used when reading the server-published global checkpoint: confirms
    the file exists AND its size has stabilized, guarding against reading
    a checkpoint the server is still mid-write on (belt-and-suspenders
    alongside the server's own atomic writes)."""
    deadline = time.time() + timeout_s
    last_size = -1
    stable_checks = 0
    while time.time() < deadline:
        if path.exists():
            size = path.stat().st_size
            if size > 0 and size == last_size:
                stable_checks += 1
                if stable_checks >= 2:
                    return True
            else:
                stable_checks = 0
            last_size = size
        time.sleep(poll_interval_s)
    return False


# --------------------------------------------------------------------------- #
# DRS resolution (GA4GH Data Repository Service)
# --------------------------------------------------------------------------- #

def resolve_drs_stream(object_id: str, drs_base_url: str) -> str:
    """Resolves a DRS ID to a local path or streamable URL via a GA4GH
    DRS-compliant resolver (e.g. the GA4GH Starter Kit)."""
    base_url = f"{drs_base_url.rstrip('/')}/ga4gh/drs/v1/objects"

    meta_resp = requests.get(f"{base_url}/{object_id}", timeout=30)
    meta_resp.raise_for_status()
    access_id = meta_resp.json()["access_methods"][0]["access_id"]

    access_url = f"{base_url}/{object_id}/access/{access_id}"
    access_resp = requests.get(access_url, timeout=30)
    access_resp.raise_for_status()
    stream_url = access_resp.json()["url"]

    if stream_url.startswith("file://"):
        return stream_url.replace("file://", "", 1)
    return stream_url


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #

PC_COLUMNS = [f"PC{i}_AVG" for i in range(1, 11)]


def load_ancestry_dataset(data_path: str) -> Tuple[TensorDataset, int]:
    """
    Loads a tab-separated coordinate table containing PC1_AVG..PC10_AVG
    feature columns and a 'super_pop' label column, returning a
    TensorDataset plus the row count (needed for sample-weighted FedAvg).
    """
    log.info("Loading local dataset from: %s", data_path)
    df = pd.read_csv(data_path, sep="\t")

    missing_cols = [c for c in PC_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected PC columns: {missing_cols}")
    if "super_pop" not in df.columns:
        raise ValueError("Dataset is missing the required 'super_pop' label column")

    unknown_labels = set(df["super_pop"].unique()) - set(SUPERPOPS)
    if unknown_labels:
        log.warning(
            "Dropping %d rows with unrecognized super_pop labels: %s",
            (df["super_pop"].isin(unknown_labels)).sum(), unknown_labels,
        )
        df = df[df["super_pop"].isin(SUPERPOPS)].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid rows remained after label filtering.")

    X = df[PC_COLUMNS].values
    y = df["super_pop"].map(SUPERPOPS.index).values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    log.info("Loaded %d samples, label distribution: %s",
              len(df), df["super_pop"].value_counts().to_dict())

    return TensorDataset(X_tensor, y_tensor), len(df)


def resolve_data_path(args: argparse.Namespace) -> str:
    """Supports either a DRS ID (resolved via the GA4GH DRS API) or a
    direct local file path, so the client can run standalone for testing
    without a DRS resolver running."""
    if args.drs_id:
        if not args.drs_base_url:
            raise ValueError("--drs-id was supplied but --drs-base-url is missing.")
        return resolve_drs_stream(args.drs_id, args.drs_base_url)
    if args.data_path:
        return args.data_path
    raise ValueError("Must supply either --drs-id (with --drs-base-url) or --data-path.")


# --------------------------------------------------------------------------- #
# Local training
# --------------------------------------------------------------------------- #

def load_global_weights(net: nn.Module, global_weights_path: Path) -> int:
    """Loads the global checkpoint produced by new_server.py into `net`.
    Returns the round number recorded in the checkpoint, for logging."""
    if not wait_for_file(global_weights_path, timeout_s=120.0):
        raise FileNotFoundError(
            f"Global weights file never appeared/stabilized: {global_weights_path}"
        )
    checkpoint = torch.load(global_weights_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    net.load_state_dict(state_dict, strict=True)
    return checkpoint.get("round", -1)


def train_local(
    net: nn.Module,
    train_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
) -> float:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    net.train()
    final_epoch_loss = 0.0
    for epoch in range(1, num_epochs + 1):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        final_epoch_loss = running_loss / max(len(train_loader), 1)
        log.info("Epoch %d/%d -- local loss: %.4f", epoch, num_epochs, final_epoch_loss)

    return final_epoch_loss


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ephemeral one-shot FL client (Flower-free, TES-task-aligned)."
    )
    parser.add_argument("--client-id", type=int, required=True)

    # Data source: either DRS-resolved or a plain local path.
    parser.add_argument("--drs-id", type=str, default=None,
                        help="DRS ID of this client's local coordinate table.")
    parser.add_argument("--drs-base-url", type=str, default="http://localhost:4500",
                        help="Base URL of the GA4GH DRS resolver, if --drs-id is used.")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Direct local path to the TSV dataset (bypasses DRS).")

    # Checkpoint I/O contract with the server.
    parser.add_argument("--global-weights-path", type=str, required=True,
                        help="Path to the global checkpoint (.pt) published by the server.")
    parser.add_argument("--output-weights-path", type=str, required=True,
                        help="Path this client must write its updated state dict to.")
    parser.add_argument("--output-meta-path", type=str, default=None,
                        help="Path for the JSON metadata sidecar (defaults to "
                             "output-weights-path with a _meta.json suffix).")
    parser.add_argument("--done-path", type=str, default=None,
                        help="Path for the completion signal flag file (defaults to "
                             "output-weights-path's sibling <client_id>.done).")

    # Model / training hyperparameters.
    parser.add_argument("--input-dim", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.005)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_weights_path = Path(args.output_weights_path)
    output_meta_path = Path(args.output_meta_path) if args.output_meta_path else (
        output_weights_path.with_name(output_weights_path.stem + "_meta.json")
    )
    done_path = Path(args.done_path) if args.done_path else (
        output_weights_path.parent / f"client_{args.client_id}.done"
    )

    log.info("Client %d starting up.", args.client_id)

    try:
        # 1. Resolve and load local dataset.
        data_path = resolve_data_path(args)
        dataset, num_samples = load_ancestry_dataset(data_path)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        # 2. Load global weights into a fresh local model instance.
        net = AncestryNet(input_dim=args.input_dim, num_classes=args.num_classes)
        round_idx = load_global_weights(net, Path(args.global_weights_path))
        log.info("Client %d loaded global weights for round %d.", args.client_id, round_idx)

        # 3. Local training.
        final_loss = train_local(
            net, train_loader,
            num_epochs=args.local_epochs,
            learning_rate=args.learning_rate,
        )

        # 4. Persist updated weights + metadata atomically.
        atomic_torch_save(
            {"round": round_idx, "client_id": args.client_id, "state_dict": net.state_dict()},
            output_weights_path,
        )
        atomic_json_save(
            {
                "client_id": args.client_id,
                "round": round_idx,
                "num_samples": num_samples,
                "local_loss": final_loss,
            },
            output_meta_path,
        )

        # 5. Signal completion LAST, only after both files are durably in place.
        #    The orchestrator polls solely on this file's existence.
        done_path.parent.mkdir(parents=True, exist_ok=True)
        done_path.write_text("ok")

        log.info(
            "Client %d finished round %d. samples=%d final_loss=%.4f -> %s",
            args.client_id, round_idx, num_samples, final_loss, output_weights_path,
        )

    except Exception as exc:
        log.error("Client %d failed: %s", args.client_id, exc)
        sys.exit(1)


if __name__ == "__main__":
    main()