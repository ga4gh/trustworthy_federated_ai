# client.py
import os
import csv
import argparse
import requests
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import AncestryNet, SUPERPOPS

parser = argparse.ArgumentParser(description="Live Two-Step GA4GH DRS Native Stateless Client")
parser.add_argument("--client-id", type=int, required=True, help="1, 2, 3, or 4")
parser.add_argument("--site-name", type=str, default=None,
                     help="Site label used in result CSVs, e.g. site_a. Defaults to 'site_<client-id>'.")
parser.add_argument("--unified-id", type=str, required=True,
                     help="DRS ID of this site's unified_data.tsv inside the Starter Kit database.")
parser.add_argument("--global-weights-path", type=str, required=True, help="Input global master checkpoint path.")
parser.add_argument("--output-weights-path", type=str, required=True, help="Target path for finalized client checkpoint.")
parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of local dataset held out for verification.")
parser.add_argument("--epochs", type=int, default=5, help="Number of local training epochs.")
parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch processing size limits.")
parser.add_argument("--lr", type=float, default=0.01, help="Adam step optimizer learning rate.")
args = parser.parse_args()

SITE_NAME = args.site_name or f"site_{args.client_id}"
RESULTS_DIR = os.path.abspath("./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CLIENT_METRICS_CSV = os.path.join(RESULTS_DIR, f"fl_client_{SITE_NAME}_metrics.csv")
FIELDNAMES = ["round", "phase", "loss", "accuracy", "num_examples"] + \
             [f"acc_{p}" for p in SUPERPOPS] + [f"n_{p}" for p in SUPERPOPS]

# Initialize or reset tracking sheets on runtime startup
if not os.path.exists(CLIENT_METRICS_CSV):
    with open(CLIENT_METRICS_CSV, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

def resolve_single_drs_stream(object_id):
    """
    Performs full two-step GA4GH lookup to resolve an exact, streamable byte URL.
    """
    meta_url = f"http://localhost:4500/ga4gh/drs/v1/objects/{object_id}"

    if object_id=="site_3_unified" or object_id == "site_4_unified":
        meta_url = f"http://localhost:4502/ga4gh/drs/v1/objects/{object_id}"
    try:
        meta_resp = requests.get(meta_url, timeout=5).json()
        access_id = meta_resp["access_methods"][0]["access_id"]
        
        access_url = f"http://localhost:4500/ga4gh/drs/v1/objects/{object_id}/access/{access_id}"
        
        if object_id=="site_3_unified" or object_id == "site_4_unified":
            access_url = f"http://localhost:4502/ga4gh/drs/v1/objects/{object_id}/access/{access_id}"
        stream_url = requests.get(access_url, timeout=5).json()["url"]
        
        if stream_url.startswith("file://"):
            stream_url = stream_url.replace("file://", "", 1)
        return stream_url
    except Exception as e:
        print(f"[DRS Error] Starter kit connection failed at {meta_url}. Falling back to clean mock local files.")
        fallback_path = f"./data_site_{args.client_id}_unified.tsv"
        if not os.path.exists(fallback_path):
            # Synthesize mockup data automatically if both DRS server and data rows are absent
            mock_data = []
            for i in range(250):
                row = {f"PC{idx}_AVG": torch.randn(1).item() for idx in range(1, 11)}
                row["super_pop"] = SUPERPOPS[i % 5]
                mock_data.append(row)
            pd.DataFrame(mock_data).to_csv(fallback_path, sep="\t", index=False)
        return fallback_path

def load_unified_dataset(unified_id):
    """
    Ingests genomics datasets completely decoupled from global structures via DRS.
    """
    print(f"\n[{SITE_NAME} Data Ingestion] Triggering two-step GA4GH DRS handshake for: {unified_id}")
    unified_stream = resolve_single_drs_stream(unified_id)
    print(f"[{SITE_NAME} Data Ingestion] Stream resolution successful -> {unified_stream}")

    df = pd.read_csv(unified_stream, sep="\t")
    
    # Isolate and sort the top 10 Eigenvector Principal Component fields cleanly
    pc_cols = [c for c in df.columns if c.upper().startswith("PC") and c.upper().endswith("_AVG")]
    pc_cols = sorted(pc_cols, key=lambda c: int(''.join(filter(str.isdigit, c))))[:10]

    X_data = df[pc_cols].values
    y_labels = df["super_pop"].apply(lambda x: SUPERPOPS.index(x)).values

    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_labels, dtype=torch.long)

    return TensorDataset(X_tensor, y_tensor)

def classwise_accuracy(y_true, y_pred):
    class_accs = {}
    class_counts = {}
    for idx, pop in enumerate(SUPERPOPS):
        mask = (y_true == idx)
        count = mask.sum().item()
        class_counts[pop] = count
        if count > 0:
            correct = ((y_pred == y_true) & mask).sum().item()
            class_accs[pop] = correct / count
        else:
            class_accs[pop] = 0.0
    return class_accs, class_counts

def main():
    # Build local compute partitions
    local_dataset = load_unified_dataset(args.unified_id)

    n_val = max(1, int(len(local_dataset) * args.val_fraction))
    n_train = len(local_dataset) - n_val
    train_subset, val_subset = random_split(
        local_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False)

    print(f"[{SITE_NAME}] Local dataset allocation: {n_train} training samples | {n_val} verification validation samples.")

    # Record site-specific sample distribution matrix footprint once up front
    _labels = local_dataset.tensors[1]
    _site_class_counts = {SUPERPOPS[c]: int((_labels == c).sum()) for c in range(len(SUPERPOPS))}
    
    sizes_csv = os.path.join(RESULTS_DIR, "fl_site_sizes.csv")
    if not os.path.exists(sizes_csv):
        with open(sizes_csv, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["site"] + SUPERPOPS + ["total"])
            
    with open(sizes_csv, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["site"] + SUPERPOPS + ["total"])
        row_data = {"site": SITE_NAME, "total": len(local_dataset)}
        row_data.update(_site_class_counts)
        writer.writerow(row_data)

    # Initialize neural network topology and read shared parameter weights dictionary
    net = AncestryNet(input_dim=10, num_classes=5)
    if not os.path.exists(args.global_weights_path):
        raise FileNotFoundError(f"Missing base master weight path definition binary: {args.global_weights_path}")
    
    global_state = torch.load(args.global_weights_path, map_location="cpu")
    # Clean up tracking dictionary if it contains embedded metadata wrappers
    if "metadata" in global_state:
        global_state = {k: v for k, v in global_state.items() if k != "metadata"}
    net.load_state_dict(global_state, strict=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # In a stateless batch computing context, we extract the target round from the output path name
    try:
        current_round = int(''.join(filter(str.isdigit, os.path.basename(args.output_weights_path).split("round_")[-1])))
    except ValueError:
        current_round = 1

    # --- Training Loop Execution Block ---
    net.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)

    # --- Stratified Evaluation Block ---
    net.eval()
    val_loss = 0.0
    correct = 0
    all_preds, all_true = [], []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = net(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
            preds = torch.max(outputs, 1)[1]
            correct += (preds == y_batch).sum().item()
            all_preds.append(preds)
            all_true.append(y_batch)

    accuracy = correct / len(val_loader.dataset)
    avg_loss = val_loss / len(val_loader.dataset)

    all_preds = torch.cat(all_preds)
    all_true = torch.cat(all_true)
    class_accs, class_counts = classwise_accuracy(all_true, all_preds)

    print(f"--> [{SITE_NAME} Round {current_round} Summary] Accuracy: {accuracy * 100:.2f}% | Metrics Loss: {avg_loss:.4f}")
    
    # Commit metrics line safely to disk repository files
    with open(CLIENT_METRICS_CSV, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        row = {
            "round": current_round, "phase": "evaluate", "loss": avg_loss,
            "accuracy": accuracy, "num_examples": len(val_subset)
        }
        row.update({f"acc_{p}": class_accs[p] for p in SUPERPOPS})
        row.update({f"n_{p}": class_counts[p] for p in SUPERPOPS})
        writer.writerow(row)

    # Output localized weights update payload along with tracking flags
    output_payload = net.state_dict()
    output_payload["metadata"] = {
        "num_examples": len(train_subset),
        "val_loss": avg_loss,
        "val_accuracy": accuracy
    }
    
    tmp_path = f"{args.output_weights_path}.tmp"
    torch.save(output_payload, tmp_path)
    os.rename(tmp_path, args.output_weights_path)
    print(f"[{SITE_NAME}] Updates successfully serialized to disk: {args.output_weights_path}")

if __name__ == "__main__":
    main()