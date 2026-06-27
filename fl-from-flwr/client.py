# client.py
import argparse
import os
import requests
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from model import AncestryNet, get_parameters, set_parameters, SUPERPOPS
from fl_metrics_logger import log_row

parser = argparse.ArgumentParser(description="Live Two-Step GA4GH DRS Native Flower Client")
parser.add_argument("--client-id", type=int, required=True, help="1, 2, 3, or 4")
parser.add_argument("--site-name", type=str, default=None,
                     help="Site label used in result CSVs, e.g. site_a. Defaults to "
                          "'site_<client-id>' if omitted.")
parser.add_argument("--genotypes-id", type=str, required=True, help="DRS ID of the sscore file")
parser.add_argument("--ancestry-id", type=str, required=True, help="DRS ID of the tsv file")
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

def load_and_align_datasets(genotypes_id, ancestry_id):
    print("\n[GA4GH DRS Orchestration] Resolving streaming vectors...")
    
    # Call the endpoints to fetch data locations
    sscore_stream = resolve_single_drs_stream(genotypes_id)
    print(f"[DRS Stream Unlocked] Genotypes destination -> {sscore_stream}")
    
    ancestry_stream = resolve_single_drs_stream(ancestry_id)
    print(f"[DRS Stream Unlocked] Ancestry destination -> {ancestry_stream}")
    
    # Read streaming endpoints or underlying files smoothly
    df_sscore = pd.read_csv(sscore_stream, sep="\t")
    df_ancestry = pd.read_csv(ancestry_stream, sep="\t")
    
    # Align matrices securely on patient identifier index (IID)
    merged_df = pd.merge(df_sscore.drop(columns=["super_pop"]), df_ancestry, on="IID")
    print(f"[Data Pipeline] Aligned matrix completely. Total unified rows: {len(merged_df)}")
    
    pc_features = [f"PC{i+1}_AVG" for i in range(10)]
    X_data = merged_df[pc_features].values
    y_labels = merged_df["super_pop"].apply(lambda x: SUPERPOPS.index(x)).values
    
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_labels, dtype=torch.long)
    
    return TensorDataset(X_tensor, y_tensor)

# Build execution set
local_dataset = load_and_align_datasets(args.genotypes_id, args.ancestry_id)
train_loader = DataLoader(local_dataset, batch_size=16, shuffle=True)

# log this site's class composition once, up front -- this is what step 3's
# size-vs-classwise-accuracy table needs per site
_labels = local_dataset.tensors[1]
_site_class_counts = {SUPERPOPS[c]: int((_labels == c).sum()) for c in range(len(SUPERPOPS))}
print(f"[{SITE_NAME}] class counts: {_site_class_counts}")
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
    site's local data get accuracy=None rather than a misleading 0.0 or NaN
    that would otherwise quietly bias the global average in plot_results.py."""
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
            for X_batch, y_batch in train_loader:
                outputs = net(X_batch)
                loss += criterion(outputs, y_batch).item()
                preds = torch.max(outputs, 1)[1]
                correct += (preds == y_batch).sum().item()
                all_preds.append(preds)
                all_true.append(y_batch)

        accuracy = correct / len(train_loader.dataset)
        avg_loss = loss / len(train_loader)

        all_preds = torch.cat(all_preds)
        all_true = torch.cat(all_true)
        class_accs, class_counts = classwise_accuracy(all_true, all_preds)

        print(f"--> [{SITE_NAME} Metrics] Accuracy: {accuracy * 100:.2f}% | Loss: {avg_loss:.4f}")
        print(f"    Classwise: " + " ".join(
            f"{p}={class_accs[p]*100:.1f}%(n={class_counts[p]})" if class_accs[p] is not None
            else f"{p}=n/a(n=0)" for p in SUPERPOPS
        ))

        _round_counter["evaluate"] += 1
        row = {"round": _round_counter["evaluate"], "phase": "evaluate", "loss": avg_loss,
               "accuracy": accuracy, "num_examples": len(train_loader.dataset)}
        row.update({f"acc_{p}": class_accs[p] for p in SUPERPOPS})
        row.update({f"n_{p}": class_counts[p] for p in SUPERPOPS})
        log_row(CLIENT_METRICS_CSV, FIELDNAMES, row)

        return float(avg_loss), len(train_loader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())
