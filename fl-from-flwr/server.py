# server.py
import flwr as fl

if __name__ == "__main__":
    print("[Aggregator Server] Waking up. Awaiting data nodes...")
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_available_clients=2,
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=15),
        strategy=strategy
    )