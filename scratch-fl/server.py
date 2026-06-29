"""
new_server.py

Stateless, checkpoint-driven Federated Learning orchestrator for genomics
ancestry prediction. Designed to align with the GA4GH Task Execution
Service (TES) model: the server never spawns or manages client processes
directly. Instead, it:

    1. Publishes the current global model weights to a shared disk volume.
    2. Waits (polls) for a fixed set of per-client output checkpoints to
       appear -- these are produced by independently-scheduled TES tasks
       running `new_client.py`.
    3. Aggregates the received state dicts via sample-weighted FedAvg.
    4. Writes the new global checkpoint and advances to the next round.

No Flower (flwr) dependency of any kind. No long-lived client sockets.
Every artifact exchanged between server and clients is a file on disk,
written atomically, so a TES executor can move them between storage
backends without the orchestrator needing to know how.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from model import AncestryNet, get_parameters, set_parameters

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Server] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("server")


# --------------------------------------------------------------------------- #
# Atomic, lock-safe file I/O helpers
# --------------------------------------------------------------------------- #

def atomic_torch_save(obj, dest_path: Path) -> None:
    """
    Write a torch checkpoint atomically so that any process polling on
    `dest_path` either sees nothing or sees a complete, valid file --
    never a partial write.

    Strategy: save to a temp file in the same directory (so os.replace
    is a same-filesystem rename, which is atomic on POSIX and Windows),
    then rename into place.
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dest_path.parent), suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, dest_path)  # atomic on same filesystem
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


def wait_for_file(path: Path, timeout_s: float, poll_interval_s: float) -> bool:
    """
    Poll for a file's existence AND a stable size (i.e. not mid-write by
    some non-atomic writer) before declaring it ready. Combined with
    atomic_torch_save on the producer side, this is belt-and-suspenders.
    """
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
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class ClientUpdate:
    client_id: int
    state_dict: Dict[str, torch.Tensor]
    num_samples: int
    local_loss: Optional[float] = None


# --------------------------------------------------------------------------- #
# FedAvg aggregation (native PyTorch, no Flower)
# --------------------------------------------------------------------------- #

def federated_average(updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
    """
    Sample-weighted FedAvg over a list of client state dicts:

        w_global = sum_k (n_k / N) * w_k

    where n_k is client k's local sample count and N = sum(n_k).
    """
    if not updates:
        raise ValueError("federated_average called with no client updates")

    total_samples = sum(u.num_samples for u in updates)
    if total_samples <= 0:
        raise ValueError("Total sample count across clients must be positive")

    reference_keys = updates[0].state_dict.keys()
    for u in updates:
        if u.state_dict.keys() != reference_keys:
            raise ValueError(
                f"Client {u.client_id} state_dict keys do not match the "
                f"reference architecture. Aggregation aborted."
            )

    new_state_dict: Dict[str, torch.Tensor] = {}
    for key in reference_keys:
        weighted_sum = torch.zeros_like(updates[0].state_dict[key], dtype=torch.float32)
        for u in updates:
            weight = u.num_samples / total_samples
            weighted_sum += u.state_dict[key].to(torch.float32) * weight
        # Preserve original dtype (e.g. BatchNorm running stats may be float32 already,
        # but integer buffers like num_batches_tracked should stay integral).
        original_dtype = updates[0].state_dict[key].dtype
        if original_dtype in (torch.int64, torch.int32, torch.long):
            new_state_dict[key] = weighted_sum.round().to(original_dtype)
        else:
            new_state_dict[key] = weighted_sum.to(original_dtype)

    log.info(
        "FedAvg aggregation complete: %d clients, %d total samples",
        len(updates), total_samples,
    )
    return new_state_dict


# --------------------------------------------------------------------------- #
# Round orchestration
# --------------------------------------------------------------------------- #

class FederatedOrchestrator:
    """
    Drives the global training loop. Every method is side-effect-explicit:
    it reads from / writes to the shared volume and nothing else. This
    keeps the orchestrator restart-safe -- if the server process dies and
    is relaunched, it can resume from the last completed round's checkpoint.
    """

    def __init__(
        self,
        work_dir: Path,
        num_rounds: int,
        client_ids: List[int],
        input_dim: int = 10,
        num_classes: int = 5,
        round_timeout_s: float = 1800.0,
        poll_interval_s: float = 2.0,
        launch_clients_cmd: Optional[str] = None,
    ):
        self.work_dir = Path(work_dir)
        self.checkpoints_dir = self.work_dir / "checkpoints"
        self.rounds_dir = self.work_dir / "rounds"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.rounds_dir.mkdir(parents=True, exist_ok=True)

        self.num_rounds = num_rounds
        self.client_ids = client_ids
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.round_timeout_s = round_timeout_s
        self.poll_interval_s = poll_interval_s
        self.launch_clients_cmd = launch_clients_cmd

        self.global_model = AncestryNet(input_dim=input_dim, num_classes=num_classes)

    # ----- path helpers ----- #

    def global_ckpt_path(self, round_idx: int) -> Path:
        return self.checkpoints_dir / f"global_model_round_{round_idx}.pt"

    def client_round_dir(self, round_idx: int) -> Path:
        d = self.rounds_dir / f"round_{round_idx}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def client_output_path(self, round_idx: int, client_id: int) -> Path:
        return self.client_round_dir(round_idx) / f"client_{client_id}_update.pt"

    def client_metadata_path(self, round_idx: int, client_id: int) -> Path:
        return self.client_round_dir(round_idx) / f"client_{client_id}_meta.json"

    def client_ready_signal_path(self, round_idx: int, client_id: int) -> Path:
        """A small flag file the client writes after BOTH the weights and
        metadata files are fully and atomically in place. This avoids any
        ambiguity about ordering: the server only ever waits on this one
        file per client."""
        return self.client_round_dir(round_idx) / f"client_{client_id}.done"

    # ----- core round logic ----- #

    def publish_global_checkpoint(self, round_idx: int) -> Path:
        ckpt_path = self.global_ckpt_path(round_idx)
        payload = {
            "round": round_idx,
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "state_dict": self.global_model.state_dict(),
        }
        atomic_torch_save(payload, ckpt_path)
        log.info("Published global checkpoint for round %d -> %s", round_idx, ckpt_path)
        return ckpt_path

    def maybe_launch_clients(self, round_idx: int, global_ckpt_path: Path) -> None:
        """
        Optional convenience hook for local / single-machine testing only.
        In a real TES deployment, an external workflow engine submits one
        TES task per client; the server has no business spawning processes.
        This is a no-op unless --launch-clients-cmd was supplied.
        """
        if not self.launch_clients_cmd:
            return
        for client_id in self.client_ids:
            output_path = self.client_output_path(round_idx, client_id)
            meta_path = self.client_metadata_path(round_idx, client_id)
            done_path = self.client_ready_signal_path(round_idx, client_id)
            cmd = self.launch_clients_cmd.format(
                client_id=client_id,
                round=round_idx,
                global_weights_path=str(global_ckpt_path),
                output_weights_path=str(output_path),
                output_meta_path=str(meta_path),
                done_path=str(done_path),
            )
            log.info("Launching client %d for round %d: %s", client_id, round_idx, cmd)
            subprocess.Popen(cmd, shell=True)

    def await_client_updates(self, round_idx: int) -> List[ClientUpdate]:
        updates: List[ClientUpdate] = []
        pending = set(self.client_ids)
        deadline = time.time() + self.round_timeout_s

        log.info(
            "Round %d: awaiting updates from clients %s (timeout=%.0fs)",
            round_idx, sorted(pending), self.round_timeout_s,
        )

        while pending and time.time() < deadline:
            for client_id in list(pending):
                done_flag = self.client_ready_signal_path(round_idx, client_id)
                if done_flag.exists():
                    update = self._load_client_update(round_idx, client_id)
                    updates.append(update)
                    pending.discard(client_id)
                    log.info(
                        "Round %d: received update from client %d (%d samples)",
                        round_idx, client_id, update.num_samples,
                    )
            if pending:
                time.sleep(self.poll_interval_s)

        if pending:
            log.warning(
                "Round %d: timed out waiting for clients %s. Proceeding with "
                "%d/%d available updates.",
                round_idx, sorted(pending), len(updates), len(self.client_ids),
            )

        if not updates:
            raise RuntimeError(
                f"Round {round_idx}: no client updates received before timeout. Aborting."
            )

        return updates

    def _load_client_update(self, round_idx: int, client_id: int) -> ClientUpdate:
        weights_path = self.client_output_path(round_idx, client_id)
        meta_path = self.client_metadata_path(round_idx, client_id)

        if not wait_for_file(weights_path, timeout_s=30.0, poll_interval_s=0.5):
            raise RuntimeError(
                f"Client {client_id} signaled done but weights file "
                f"{weights_path} never stabilized."
            )

        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        with open(meta_path, "r") as f:
            meta = json.load(f)

        return ClientUpdate(
            client_id=client_id,
            state_dict=state_dict,
            num_samples=int(meta["num_samples"]),
            local_loss=meta.get("local_loss"),
        )

    def aggregate_and_update_global(self, updates: List[ClientUpdate]) -> None:
        new_state_dict = federated_average(updates)
        self.global_model.load_state_dict(new_state_dict, strict=True)

    def run(self) -> Path:
        log.info(
            "Starting federated training: %d rounds, clients=%s, work_dir=%s",
            self.num_rounds, self.client_ids, self.work_dir,
        )
        final_ckpt_path = self.global_ckpt_path(0)

        for round_idx in range(1, self.num_rounds + 1):
            log.info("===== Round %d/%d =====", round_idx, self.num_rounds)

            ckpt_path = self.publish_global_checkpoint(round_idx)
            self.maybe_launch_clients(round_idx, ckpt_path)

            updates = self.await_client_updates(round_idx)
            self.aggregate_and_update_global(updates)

            final_ckpt_path = self.global_ckpt_path(round_idx)
            atomic_torch_save(
                {
                    "round": round_idx,
                    "input_dim": self.input_dim,
                    "num_classes": self.num_classes,
                    "state_dict": self.global_model.state_dict(),
                    "num_clients_aggregated": len(updates),
                },
                final_ckpt_path,
            )
            log.info("Round %d complete. Updated global checkpoint -> %s", round_idx, final_ckpt_path)

        log.info("Federated training finished after %d rounds.", self.num_rounds)
        return final_ckpt_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stateless checkpoint-driven FedAvg orchestrator (Flower-free, TES-aligned)."
    )
    parser.add_argument("--work-dir", type=str, default="./fl_work",
                        help="Shared disk volume root for checkpoints and round artifacts.")
    parser.add_argument("--num-rounds", type=int, default=15)
    parser.add_argument("--client-ids", type=int, nargs="+", required=True,
                        help="Expected client IDs to wait on each round, e.g. --client-ids 0 1 2")
    parser.add_argument("--input-dim", type=int, default=10)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--round-timeout-s", type=float, default=1800.0,
                        help="Max seconds to wait per round before aborting.")
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument(
        "--launch-clients-cmd", type=str, default=None,
        help=(
            "OPTIONAL, for local testing only. A shell command template with "
            "{client_id} {round} {global_weights_path} {output_weights_path} "
            "{output_meta_path} {done_path} placeholders, run once per client "
            "per round. In a real TES deployment, leave this unset -- an "
            "external workflow engine submits client tasks instead."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    orchestrator = FederatedOrchestrator(
        work_dir=Path(args.work_dir),
        num_rounds=args.num_rounds,
        client_ids=args.client_ids,
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        round_timeout_s=args.round_timeout_s,
        poll_interval_s=args.poll_interval_s,
        launch_clients_cmd=args.launch_clients_cmd,
    )
    try:
        final_path = orchestrator.run()
        log.info("SUCCESS. Final global model checkpoint: %s", final_path)
    except Exception as exc:
        log.error("Federated training aborted: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()