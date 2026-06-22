import argparse
import requests
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from model import AncestryNet, get_parameters, set_parameters, SUPERPOPS

parser = argparse.ArgumentParser(description="Federated Client using GA4GH DRS")
parser.add_argument("--client-id", type=int, required=True)
parser.add_argument("--drs-id", type=str, required=True, help="DRS ID of the unified file")
args = parser.parse_args()

def resolve_drs_stream(object_id):
    """Resolves a DRS ID to a streaming URL via the local GA4GH Starter Kit."""
    base_url = "http://localhost:4500/ga4gh/drs/v1/objects"
    
    # 1. Fetch metadata to catch access method mapping
    meta_resp = requests.get(f"{base_url}/{object_id}").json()
    access_id = meta_resp["access_methods"][0]["access_id"]
    
    # 2. Extract direct streaming link
    access_url = f"{base_url}/{object_id}/access/{access_id}"
    stream_url = requests.get(access_url).json()["url"]
    
    if stream_url.startswith("file://"):
        return stream_url.replace("file://", "", 1)
    return stream_url

def load_unified_dataset(drs_id):
    print(f"\n[DRS] Resolving unified dataset: {drs_id}")
    stream_path = resolve_drs_stream(drs_id)
    
    # Load the TSV tracking real coordinates layout
    df = pd.read_csv(stream_path, sep="\t")
    
    # EXACT COLUMN FILTERING: Targets exactly PC1_AVG through PC10_AVG
    pc_features = [f"PC{i}_AVG" for i in range(1, 11)]
    X_data = df[pc_features].values
    
    # Match strings to index positions securely (0-4)
    y_labels = df["super_pop"].apply(lambda x: SUPERPOPS.index(x) if x in SUPERPOPS else 0).values
    
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_labels, dtype=torch.long)
    
    return TensorDataset(X_tensor, y_tensor)

# Setup infrastructure
local_dataset = load_unified_dataset(args.drs_id)
train_loader = DataLoader(local_dataset, batch_size=32, shuffle=True)

# Explicitly instantiate network model architecture dimension boundaries to 10
net = AncestryNet(input_dim=10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

class Client(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_parameters(net)

    def fit(self, parameters, config):
        set_parameters(net, parameters)
        net.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(net(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return get_parameters(net), len(local_dataset), {"loss": running_loss / len(train_loader)}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        net.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                outputs = net(X_batch)
                loss += criterion(outputs, y_batch).item()
                correct += (torch.max(outputs, 1)[1] == y_batch).sum().item()
        accuracy = correct / len(local_dataset)
        return float(loss / len(train_loader)), len(local_dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())