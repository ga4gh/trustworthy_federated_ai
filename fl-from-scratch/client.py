# client.py
import argparse
import requests
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from model import AncestryMLP, get_parameters, set_parameters

parser = argparse.ArgumentParser(description="Live Two-Step GA4GH DRS Native Flower Client")
parser.add_argument("--client-id", type=int, required=True, help="1 or 2")
parser.add_argument("--drs-uri", type=str, required=True, help="drs://localhost:4500/mock_1k_genomes_nodeX")
args = parser.parse_args()

def fetch_data_from_drs_compliant(drs_uri):
    """
    Executes a fully specification-compliant GA4GH DRS two-step resolution sequence.
    Step 1: Get metadata object to extract access_id.
    Step 2: Request the raw URL link using that specific access block token.
    """
    print(f"\n[GA4GH DRS] Intercepted protocol lookup URI: {drs_uri}")
    object_id = drs_uri.split("/")[-1]
    
    # --- STEP 1: Metadata Extraction ---
    drs_metadata_url = f"http://localhost:4500/ga4gh/drs/v1/objects/{object_id}"
    print(f"[GA4GH DRS] Step 1: Requesting metadata from public portal...")
    
    response = requests.get(drs_metadata_url)
    response.raise_for_status()
    metadata = response.json()
    
    # Safely extract access ID token from the top structure
    access_method = metadata["access_methods"][0]
    access_id = access_method["access_id"]
    print(f"[GA4GH DRS] Step 1 Complete. Dynamic Access ID acquired: '{access_id}'")
    
    # --- STEP 2: Access Token Exchange ---
    drs_access_url = f"http://localhost:4500/ga4gh/drs/v1/objects/{object_id}/access/{access_id}"
    print(f"[GA4GH DRS] Step 2: Requesting authorization data stream pointer...")
    
    access_response = requests.get(drs_access_url)
    access_response.raise_for_status()
    access_data = access_response.json()
    
    # The Starter Kit returns a direct dictionary containing the true file url or path mapping
    resolved_file_url = access_data["url"]
    print(f"[GA4GH DRS] Step 2 Complete. Storage address unlocked: {resolved_file_url}")
    
    # Clean up standard 'file://' prefix if present for native pandas parsing compatibility
    if resolved_file_url.startswith("file://"):
        resolved_file_url = resolved_file_url.replace("file://", "", 1)
        
    print(f"[GA4GH DRS] Handshake finished. Loading genetic tracking vector directly into memory...")
    
    # Load the synthetic 1000 Genomes Principal Component coordinates from the target CSV
    df = pd.read_csv(resolved_file_url)
    X = torch.tensor(df.iloc[:, :20].values, dtype=torch.float32)
    y = torch.tensor(df["superpop"].values, dtype=torch.long)
    
    print(f"[GA4GH DRS] Matrix Ingestion Success. Dimensions: Features={X.shape}, Classes={len(y.unique())}\n")
    return TensorDataset(X, y)

# Initialize dataset directly from the secure two-step protocol resolution
local_dataset = fetch_data_from_drs_compliant(args.drs_uri)
train_loader = DataLoader(local_dataset, batch_size=32, shuffle=True)

# Define network core runtime components
net = AncestryMLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

class TrustworthyClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_parameters(net)

    def fit(self, parameters, config):
        set_parameters(net, parameters)
        net.train()
        for epoch in range(5):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                loss = criterion(net(X_batch), y_batch)
                loss.backward()
                optimizer.step()
        return get_parameters(net), len(train_loader.dataset), {}

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
        print(f"[Client {args.client_id} Evaluation] Round Completed. Current Accuracy: {accuracy:.4f}")
        return float(loss/len(train_loader)), len(train_loader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=TrustworthyClient())