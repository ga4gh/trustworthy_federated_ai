# client.py
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
from model import SimpleMLP, get_parameters, set_parameters

# 1. Parse command-line argument to distinguish multiple mock clients
parser = argparse.ArgumentParser(description="Flower Client")
parser.add_argument("--client-id", type=int, required=True, help="ID of the client (1 or 2)")
args = parser.parse_args()

# 2. Mock isolated local data (Imagine this is reading local files securely)
# Input features: 20-dimensional coordinates (like compressed PCA eigenvectors)
if args.client_id == 1:
    X_local = torch.randn(200, 20)
    y_local = torch.randint(0, 5, (200,))
else:
    X_local = torch.randn(150, 20)
    y_local = torch.randint(0, 5, (150,))

dataset = TensorDataset(X_local, y_local)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Initialize local model framework
net = SimpleMLP(input_dim=20, num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# 4. Define the Flower Client logic
class AncestryClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_parameters(net)

    def fit(self, parameters, config):
        print(f"[Client {args.client_id}] Received global model weights. Starting local epoch...")
        set_parameters(net, parameters)
        
        # Local training loop execution
        net.train()
        for epoch in range(2):  # Local epochs execution
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = net(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
        print(f"[Client {args.client_id}] Local training complete. Transmitting math arrays back to server.")
        return get_parameters(net), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        net.eval()
        # For evaluation showcase, computing loss on local data slice
        loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in train_loader:
                outputs = net(X_batch)
                loss += criterion(outputs, y_batch).item()
                correct += (torch.max(outputs, 1)[1] == y_batch).sum().item()
        
        accuracy = correct / len(train_loader.dataset)
        return float(loss/len(train_loader)), len(train_loader.dataset), {"accuracy": float(accuracy)}

# 5. Connect and start client lifecycle
if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=AncestryClient())