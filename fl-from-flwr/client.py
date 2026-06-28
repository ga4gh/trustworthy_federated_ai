# client.py
import argparse
import os
import requests
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import flwr as fl
from model import AncestryNet, get_parameters, set_parameters, SUPERPOPS
from fl_metrics_logger import log_row

parser = argparse.ArgumentParser(description="Live Two-Step GA4GH DRS Native Flower Client")
parser.add_argument("--client-id", type=int, required=True, help="1, 2, 3, or 4")
parser.add_argument("--site-name", type=str, default=None,
                     help="Site label used in result CSVs, e.g. site_a. Defaults to "
                          "'site_<client-id>' if omitted.")
parser.add_argument("--unified-id", type=str, required=True,
                     help="DRS ID of this site's unified_data.tsv (as produced by "
                          "merge_flan_data_4sites.py -- genotypes + ancestry already "
                          "merged into one file, labeled with a 'super_pop' column).")
parser.add_argument("--val-fraction", type=float, default=0.2,
                     help="Fraction of this site's local data held out for evaluation. "
                          "Evaluating on the same rows used for training (the previous "
                          "behavior) reports training accuracy, not generalization -- "
                          "with only a few hundred points per site that hits ~100% almost "
                          "immediately and tells you nothing.")
args = parser.parse_args()

SITE_NAME = args.site_name or f"site_{args.client_id}"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
CLIENT_METRICS_CSV = os.path.join(RESULTS_DIR, f"fl_client_{SITE_NAME}_metrics.csv")
FIELDNAMES = ["round", "phase", "loss", "accuracy", "num_examples"] + \
             [f"acc_{p}" for p in SUPERPOPS] + [f"n_{p}" for p in SUPERPOPS]

# reset this client's CSV at process start, same reasoning as server.py
if os.path.exists(CLIENT_METRICS_CSV):
    os.remove(CLIENT_METRICS_CSV)

_round_counter = {"fit": 0, "evaluate": 0}


def resolve_single_drs_stream(object_id):
    """
    Performs full two-step GA4GH lookup to resolve an exact, streamable byte URL.
    """
    # Step 1: Retrieve Access ID from metadata
    meta_url = f"http://localhost:4500/ga4gh/drs/v1/objects/{object_id}"
    meta_resp = requests.get(meta_url).json()
    access_id = meta_resp["access_methods"][0]["access_id"]
    
    # Step 2: Acquire tokenized stream pointer
    access_url = f"http://localhost:4500/ga4gh/drs/v1/objects/{object_id}/access/{access_id}"
    stream_url = requests.get(access_url).json()["url"]
    
    # Clean up standard 'file://' prefix if the starter kit fallback mode exposes it directly
    if stream_url.startswith("file://"):
        stream_url = stream_url.replace("file://", "", 1)
        
    return stream_url

def load_unified_dataset(unified_id):
    """unified_data.tsv (from merge_flan_data_4sites.py) is already merged --
    one row per individual, PC columns + a 'super_pop' label column. No second
    DRS object, no client-side pd.merge needed."""
    print("\n[GA4GH DRS Orchestration] Resolving streaming vector...")

    unified_stream = resolve_single_drs_stream(unified_id)
    print(f"[DRS Stream Unlocked] Unified data destination -> {unified_stream}")

    df = pd.read_csv(unified_stream, sep="\t")
    print(f"[Data Pipeline] Loaded unified site data. Total rows: {len(df)}")

    pc_cols = [c for c in df.columns if c.upper().startswith("PC") and c.upper().endswith("_AVG")]
    pc_cols = sorted(pc_cols, key=lambda c: int(''.join(filter(str.isdigit, c))))

    X_data = df[pc_cols].values
    # merge_flan_data_4sites.py writes the label column as 'super_pop', not 'superpop'
    y_labels = df["super_pop"].apply(lambda x: SUPERPOPS.index(x)).values

    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_labels, dtype=torch.long)

    return TensorDataset(X_tensor, y_tensor)

# Build execution set
local_dataset = load_unified_dataset(args.unified_id)

# Real held-out split: train on train_subset, evaluate (and report classwise
# accuracy) ONLY on val_subset. Without this, evaluate() below would just be
# scoring the model on rows it already memorized during fit().
n_val = max(1, int(len(local_dataset) * args.val_fraction))
n_train = len(local_dataset) - n_val
train_subset, val_subset = random_split(
    local_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
print(f"[{SITE_NAME}] local split: {n_train} train / {n_val} val individuals")

# log this site's class composition once, up front -- this is what step 3's
# size-vs-classwise-accuracy table needs per site. Logged on the FULL local
# dataset (train+val combined) since that's the site's true population.
_labels = local_dataset.tensors[1]
_site_class_counts = {SUPERPOPS[c]: int((_labels == c).sum()) for c in range(len(SUPERPOPS))}
print(f"[{SITE_NAME}] class counts (train+val): {_site_class_counts}")
log_row(
    os.path.join(RESULTS_DIR, "fl_site_sizes.csv"),
    ["site"] + SUPERPOPS + ["total"],
    {"site": SITE_NAME, **_site_class_counts, "total": len(local_dataset)},
)

net = AncestryNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)


def classwise_accuracy(y_true, y_pred):
    """Per-superpopulation accuracy + support count. Classes absent from this
    site's VAL split get accuracy=None rather than a misleading 0.0 or NaN
    that would otherwise quietly bias the global average in plot_results.py.
    With small per-site val splits, a class can easily have zero val examples
    even if it's present in the site's training data."""
    accs, counts = {}, {}
    for c, pop in enumerate(SUPERPOPS):
        mask = y_true == c
        n = int(mask.sum())
        counts[pop] = n
        accs[pop] = float((y_pred[mask] == c).float().mean()) if n > 0 else None
    return accs, counts


class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_parameters(net)

    def fit(self, parameters, config):
        set_parameters(net, parameters)
        net.train()
        running_loss = 0.0
        for epoch in range(3):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(net(X_batch), y_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        _round_counter["fit"] += 1
        avg_loss = running_loss / len(train_loader)

        row = {"round": _round_counter["fit"], "phase": "fit", "loss": avg_loss,
               "accuracy": None, "num_examples": len(train_loader.dataset)}
        row.update({f"acc_{p}": None for p in SUPERPOPS})
        row.update({f"n_{p}": None for p in SUPERPOPS})
        log_row(CLIENT_METRICS_CSV, FIELDNAMES, row)

        return get_parameters(net), len(train_loader.dataset), {"loss": avg_loss}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        net.eval()
        loss, correct = 0.0, 0
        all_preds, all_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = net(X_batch)
                loss += criterion(outputs, y_batch).item()
                preds = torch.max(outputs, 1)[1]
                correct += (preds == y_batch).sum().item()
                all_preds.append(preds)
                all_true.append(y_batch)

        accuracy = correct / len(val_loader.dataset)
        avg_loss = loss / len(val_loader)

        all_preds = torch.cat(all_preds)
        all_true = torch.cat(all_true)
        class_accs, class_counts = classwise_accuracy(all_true, all_preds)

        print(f"--> [{SITE_NAME} Metrics, held-out val] Accuracy: {accuracy * 100:.2f}% | Loss: {avg_loss:.4f}")
        print(f"    Classwise: " + " ".join(
            f"{p}={class_accs[p]*100:.1f}%(n={class_counts[p]})" if class_accs[p] is not None
            else f"{p}=n/a(n=0)" for p in SUPERPOPS
        ))

        _round_counter["evaluate"] += 1
        row = {"round": _round_counter["evaluate"], "phase": "evaluate", "loss": avg_loss,
               "accuracy": accuracy, "num_examples": len(val_loader.dataset)}
        row.update({f"acc_{p}": class_accs[p] for p in SUPERPOPS})
        row.update({f"n_{p}": class_counts[p] for p in SUPERPOPS})
        log_row(CLIENT_METRICS_CSV, FIELDNAMES, row)

        return float(avg_loss), len(val_loader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())