# plot_results.py
"""
Reads:
    results/baseline_metrics.csv
    results/fl_server_metrics.csv
    results/fl_site_sizes.csv
    results/fl_client_<site>_metrics.csv   (one per site, written by client.py)

Produces:
    results/table_loss_accuracy.csv     summary table: per-site size + final accuracy
    results/comparison_curve.png        baseline vs FL loss & accuracy over training
    results/classwise_accuracy.png      per-superpop accuracy bars + size correlation

Run this after baseline_train.py and a full FL run (server.py + 4x client.py)
have both finished and left their CSVs in results/.
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SUPERPOPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]


def load_baseline():
    path = os.path.join(RESULTS_DIR, "baseline_metrics.csv")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found -- run baseline_train.py first. Skipping baseline curves.")
        return None
    return pd.read_csv(path)


def load_fl_server():
    path = os.path.join(RESULTS_DIR, "fl_server_metrics.csv")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found -- run server.py + client.py first. Skipping FL curves.")
        return None
    return pd.read_csv(path)


def load_site_sizes():
    path = os.path.join(RESULTS_DIR, "fl_site_sizes.csv")
    if not os.path.exists(path):
        print(f"WARNING: {path} not found. Skipping site-size table.")
        return None
    df = pd.read_csv(path)
    # client.py logs its size row every process start; if a client was restarted
    # keep only its most recent row per site
    return df.drop_duplicates(subset="site", keep="last").reset_index(drop=True)


def load_client_classwise_latest():
    """For each site's client CSV, take the most recent 'evaluate' row -- that's
    the final-round classwise accuracy after the last aggregated model update."""
    pattern = os.path.join(RESULTS_DIR, "fl_client_*_metrics.csv")
    rows = []
    for path in sorted(glob.glob(pattern)):
        site = os.path.basename(path).replace("fl_client_", "").replace("_metrics.csv", "")
        df = pd.read_csv(path)
        eval_rows = df[df["phase"] == "evaluate"]
        if eval_rows.empty:
            continue
        last = eval_rows.iloc[-1]
        row = {"site": site, "round": last["round"], "accuracy": last["accuracy"]}
        for p in SUPERPOPS:
            row[f"acc_{p}"] = last.get(f"acc_{p}")
            row[f"n_{p}"] = last.get(f"n_{p}")
        rows.append(row)
    if not rows:
        print("WARNING: no fl_client_*_metrics.csv files found. Skipping classwise plot.")
        return None
    return pd.DataFrame(rows)


def plot_comparison_curve(baseline_df, fl_server_df):
    if baseline_df is None and fl_server_df is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    if baseline_df is not None:
        axes[0].plot(baseline_df["epoch"], baseline_df["val_loss"],
                     label="baseline (centralized)", linewidth=2)
        axes[1].plot(baseline_df["epoch"], baseline_df["val_acc"],
                     label="baseline (centralized)", linewidth=2)

    if fl_server_df is not None:
        # x-axis unit differs (FL round vs baseline epoch); plot FL on its own
        # round index so the two convergence shapes are still visually comparable
        axes[0].plot(fl_server_df["round"], fl_server_df["agg_loss"],
                     label="FL (aggregated)", linewidth=2, linestyle="--")
        axes[1].plot(fl_server_df["round"], fl_server_df["agg_accuracy"],
                     label="FL (aggregated)", linewidth=2, linestyle="--")

    axes[0].set_title("Loss: Baseline vs Federated")
    axes[0].set_xlabel("epoch (baseline) / round (FL)")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].set_title("Accuracy: Baseline vs Federated")
    axes[1].set_xlabel("epoch (baseline) / round (FL)")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend()

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "comparison_curve.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")


def build_size_accuracy_table(site_sizes_df, classwise_df):
    if site_sizes_df is None or classwise_df is None:
        print("Skipping table_loss_accuracy.csv (missing site sizes or classwise accuracy).")
        return None

    merged = site_sizes_df.merge(classwise_df, on="site", how="inner")
    cols = ["site", "total", "accuracy"] + SUPERPOPS + [f"acc_{p}" for p in SUPERPOPS]
    table = merged[[c for c in cols if c in merged.columns]]

    out_path = os.path.join(RESULTS_DIR, "table_loss_accuracy.csv")
    table.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    print("\n" + table.to_string(index=False))
    return table


def plot_classwise_accuracy(classwise_df, site_sizes_df):
    if classwise_df is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # left: grouped bar chart, per-site classwise accuracy
    x = np.arange(len(SUPERPOPS))
    width = 0.8 / max(len(classwise_df), 1)
    for i, (_, row) in enumerate(classwise_df.iterrows()):
        accs = [row.get(f"acc_{p}") for p in SUPERPOPS]
        accs = [a if pd.notna(a) else 0 for a in accs]
        axes[0].bar(x + i * width, accs, width=width, label=row["site"])
    axes[0].set_xticks(x + width * (len(classwise_df) - 1) / 2)
    axes[0].set_xticklabels(SUPERPOPS)
    axes[0].set_ylabel("accuracy")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_title("Classwise Accuracy per Site")
    axes[0].legend(fontsize=8)

    # right: does class SIZE (pooled across sites) predict classwise accuracy?
    if site_sizes_df is not None:
        pooled_size = site_sizes_df[SUPERPOPS].sum()
        # average each class's accuracy across sites that actually had that class present
        pooled_acc = {}
        for p in SUPERPOPS:
            vals = classwise_df[f"acc_{p}"].dropna()
            pooled_acc[p] = vals.mean() if not vals.empty else np.nan

        sizes = [pooled_size[p] for p in SUPERPOPS]
        accs = [pooled_acc[p] for p in SUPERPOPS]
        axes[1].scatter(sizes, accs, s=80)
        for p, sx, sy in zip(SUPERPOPS, sizes, accs):
            axes[1].annotate(p, (sx, sy), textcoords="offset points", xytext=(5, 5))
        axes[1].set_xlabel("total individuals in this class (pooled across sites)")
        axes[1].set_ylabel("mean classwise accuracy")
        axes[1].set_title("Class Size vs Accuracy")
        axes[1].set_ylim(0, 1.0)

        valid = [(sx, sy) for sx, sy in zip(sizes, accs) if pd.notna(sy)]
        if len(valid) >= 2:
            xs, ys = zip(*valid)
            corr = np.corrcoef(xs, ys)[0, 1]
            axes[1].text(0.05, 0.05, f"corr = {corr:.2f}", transform=axes[1].transAxes)

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "classwise_accuracy.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved -> {out_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    baseline_df = load_baseline()
    fl_server_df = load_fl_server()
    site_sizes_df = load_site_sizes()
    classwise_df = load_client_classwise_latest()

    plot_comparison_curve(baseline_df, fl_server_df)
    build_size_accuracy_table(site_sizes_df, classwise_df)
    plot_classwise_accuracy(classwise_df, site_sizes_df)


if __name__ == "__main__":
    main()
