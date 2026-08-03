# generate_mock_data.py
import os
import torch
import pandas as pd

# Create a local directory for our DRS endpoints to point to
os.makedirs("./drs_data", exist_ok=True)

def create_node_dataset(file_path, client_id, num_samples=300):
    torch.manual_seed(client_id)
    
    # Randomly assign ancestral lines across the 5 global superpopulations
    labels = torch.randint(0, 5, (num_samples,))
    features = torch.randn(num_samples, 20)
    
    # Inject population shifts so the neural network can actually learn clusters
    for i in range(num_samples):
        target_pop = labels[i].item()
        features[i, :5] += target_pop * 2.0  # Simulated ancestral variation axes
    
    columns = [f"PC{i+1}" for i in range(20)] + ["superpop"]
    data = torch.cat([features, labels.unsqueeze(1).float()], dim=1).numpy()
    df = pd.DataFrame(data, columns=columns)
    
    df.to_csv(file_path, index=False)
    print(f"Generated dataset at: {file_path} ({num_samples} samples)")

create_node_dataset("./drs_data/node1_coordinates.csv", client_id=1)
create_node_dataset("./drs_data/node2_coordinates.csv", client_id=2)