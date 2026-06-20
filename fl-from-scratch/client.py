# client.py
import argparse
import requests
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from model import AncestryNet, get_parameters, set_parameters, SUPERPOPS

parser = argparse.ArgumentParser(description="Live Two-Step GA4GH DRS Native Flower Client")
parser.add_argument("--client-id", type=int, required=True, help="1 or 2")
parser.add_argument("--genotypes-id", type=str, required=True, help="DRS ID of the sscore file")
parser.add_argument("--ancestry-id", type=str, required=True, help="DRS ID of the tsv file")
args = parser.parse_args()

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
    
    # Align matrices securely on patient identifier index (#IID)
    merged_df = pd.merge(df_sscore, df_ancestry, on="#IID")
    print(f"[Data Pipeline] Aligned matrix completely. Total unified rows: {len(merged_df)}")
    
    pc_features = [f"PC{i+1}_Avg" for i in range(20)]
    X_data = merged_df[pc_features].values
    y_labels = merged_df["superpop"].apply(lambda x: SUPERPOPS.index(x)).values
    
    X_tensor = torch.tensor(X_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_labels, dtype=torch.long)
    
    return TensorDataset(X_tensor, y_tensor)

# Build execution set
local_dataset = load_and_align_datasets(args.genotypes_id, args.ancestry_id)
train_loader = DataLoader(local_dataset, batch_size=16, shuffle=True)

net = AncestryNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.005)

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
        return get_parameters(net), len(train_loader.dataset), {"loss": running_loss / len(train_loader)}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        net.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                outputs = net(X_batch)
                loss += criterion(outputs, y_batch).item()
                correct += (torch.max(outputs, 1)[1] == y_batch).sum().item()
        
        accuracy = correct / len(train_loader.dataset)
        print(f"--> [Node {args.client_id} Metrics] Accuracy: {accuracy * 100:.2f}%")
        return float(loss/len(train_loader)), len(train_loader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())