# server.py
import flwr as fl

if __name__ == "__main__":
    print("Initializing central aggregator. Awaiting distributed clients...")
    
    # Define standard Federated Averaging strategy requiring minimum 2 clients to check-in
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,       # Minimum clients sampled for a training round
        min_available_clients=2, # Minimum clients that must check in before triggering Round 1
    )
    
    # Launch network listening socket
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=30), # Federated learning lifecycle round counter
        strategy=strategy
    )