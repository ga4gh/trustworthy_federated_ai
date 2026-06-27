# fl_metrics_logger.py
"""
Minimal append-only CSV logger shared by client.py and server.py so every
round's loss AND accuracy land in one place that plot_results.py can read.
Avoids needing a database or extra service: just a flat file per role.
"""
import csv
import os
import threading

_lock = threading.Lock()


def log_row(csv_path, fieldnames, row: dict):
    """Append one row to csv_path, writing the header if the file is new."""
    with _lock:
        file_exists = os.path.isfile(csv_path)
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
