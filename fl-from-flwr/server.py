# server.py
import os
import flwr as fl
from fl_metrics_logger import log_row

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SERVER_METRICS_CSV = os.path.join(RESULTS_DIR, "fl_server_metrics.csv")
FIELDNAMES = ["round", "agg_loss", "agg_accuracy", "num_clients"]


def weighted_average(metrics):
    """
    metrics: list of (num_examples, metrics_dict) tuples, one per client,
    as returned by Client.evaluate(). FedAvg's default aggregation only
    averages the scalar loss it gets back from evaluate() -- it never
    looks inside the metrics dict, so 'accuracy' silently never reaches
    the console unless we aggregate it ourselves here.
    """
    total_examples = sum(n for n, _ in metrics)
    agg_accuracy = sum(n * m.get("accuracy", 0.0) for n, m in metrics) / total_examples
    return {"accuracy": agg_accuracy}


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that also writes round-by-round loss+accuracy to a CSV file
    as soon as aggregate_evaluate runs, so plot_results.py has something
    to read without parsing stdout."""

    def aggregate_evaluate(self, server_round, results, failures):
        agg_loss, agg_metrics = super().aggregate_evaluate(server_round, results, failures)
        agg_accuracy = agg_metrics.get("accuracy") if agg_metrics else None

        loss_str = f"{agg_loss:.4f}" if agg_loss is not None else "n/a"
        acc_str = f"{agg_accuracy*100:.2f}%" if agg_accuracy is not None else "n/a"
        print(f"[Round {server_round}] aggregated loss={loss_str} accuracy={acc_str}")

        log_row(
            SERVER_METRICS_CSV,
            FIELDNAMES,
            {
                "round": server_round,
                "agg_loss": agg_loss,
                "agg_accuracy": agg_accuracy,
                "num_clients": len(results),
            },
        )
        return agg_loss, agg_metrics


if __name__ == "__main__":
    print("[Aggregator Server] Waking up. Awaiting data nodes...")

    # fresh metrics file each run so plot_results.py never mixes old + new rounds
    if os.path.exists(SERVER_METRICS_CSV):
        os.remove(SERVER_METRICS_CSV)

    strategy = LoggingFedAvg(
        min_fit_clients=4,
        min_available_clients=4,
        min_evaluate_clients=4,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=15),
        strategy=strategy,
    )
