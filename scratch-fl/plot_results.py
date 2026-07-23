# plot_results.py
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

CHECKPOINTS_DIR = "./checkpoints"
RESULTS_DIR     = "./results"

SUPERPOPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]

C_FL_LOSS   = "#d63031"
C_FL_ACC    = "#0984e3"
C_BL_LOSS   = "#e17055"
C_BL_ACC    = "#00b894"
C_GRID      = "#dfe6e9"
HMAP_COLORS = ["#ffffff", "#74b9ff", "#0984e3", "#2d3436"]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Original FL-only convergence plot
# ══════════════════════════════════════════════════════════════════════════════
def generate_performance_plots(metrics_csv_path: str, output_image_path: str):
    if not os.path.exists(metrics_csv_path):
        print(f"[-] Server metrics not found: {metrics_csv_path}")
        return
    df = pd.read_csv(metrics_csv_path)
    if df.empty:
        print(f"[-] Metrics CSV is empty.")
        return

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()

    ax1.set_xlabel("Global Federated Rounds", fontsize=12, fontweight="bold", labelpad=10)
    ax1.set_ylabel("Global Weighted Loss", color=C_FL_LOSS, fontsize=12, fontweight="bold")
    ax1.plot(df["round"], df["weighted_loss"], color=C_FL_LOSS, marker="o",
             markersize=6, linewidth=2, label="Federated Loss")
    ax1.tick_params(axis="y", labelcolor=C_FL_LOSS)
    ax1.grid(True, linestyle="--", alpha=0.5)

    acc_vals = df["weighted_accuracy"] if df["weighted_accuracy"].max() > 1.0 else df["weighted_accuracy"] * 100
    ax2.set_ylabel("Global Weighted Accuracy (%)", color=C_FL_ACC, fontsize=12, fontweight="bold")
    ax2.plot(df["round"], acc_vals, color=C_FL_ACC, marker="s",
             markersize=6, linewidth=2, label="Federated Accuracy")
    ax2.tick_params(axis="y", labelcolor=C_FL_ACC)

    lines = [plt.Line2D([0], [0], color=C_FL_LOSS, lw=2, marker="o", label="Federated Loss"),
             plt.Line2D([0], [0], color=C_FL_ACC,  lw=2, marker="s", label="Federated Accuracy")]
    ax1.legend(handles=lines, loc="center right", frameon=True, facecolor="white", edgecolor="#b2bec3")
    ax1.set_xticks(df["round"])

    plt.title(
        "Trustworthy Federated AI: Genetic Ancestry Convergence Analytics\n"
        "(GA4GH Checkpoint-Driven Stateless Architecture Baseline)",
        fontsize=13, fontweight="bold", pad=15
    )
    fig.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] FL convergence plot → {output_image_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. FL vs Baseline convergence comparison
# ══════════════════════════════════════════════════════════════════════════════
def plot_fl_vs_baseline(fl_csv: str, baseline_csv: str, out_path: str):
    if not os.path.exists(fl_csv):
        print(f"[-] FL metrics not found: {fl_csv}")
        return
    if not os.path.exists(baseline_csv):
        print(f"[-] Baseline metrics not found: {baseline_csv}  (run baseline_train.py first)")
        return

    fl = pd.read_csv(fl_csv)
    bl = pd.read_csv(baseline_csv)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(fl["round"], fl["weighted_loss"],
             color=C_FL_LOSS, marker="o", markersize=5, linewidth=2, label="FL Loss")
    ax1.plot(bl["round"], bl["weighted_loss"],
             color=C_BL_LOSS, marker="^", markersize=5, linewidth=2, linestyle="--", label="Baseline Loss")
    ax1.set_xlabel("Round (FL) / Equivalent Round (Baseline)", fontsize=11, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Weighted Loss", color=C_FL_LOSS, fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=C_FL_LOSS)
    ax1.grid(True, linestyle="--", alpha=0.4, color=C_GRID)

    def pct(s):
        return s if s.max() > 1.0 else s * 100

    ax2.plot(fl["round"], pct(fl["weighted_accuracy"]),
             color=C_FL_ACC, marker="s", markersize=5, linewidth=2, label="FL Accuracy")
    ax2.plot(bl["round"], pct(bl["weighted_accuracy"]),
             color=C_BL_ACC, marker="D", markersize=5, linewidth=2, linestyle="--", label="Baseline Accuracy")
    ax2.set_ylabel("Weighted Accuracy (%)", color=C_FL_ACC, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=C_FL_ACC)

    lines = [
        plt.Line2D([0], [0], color=C_FL_LOSS, lw=2, marker="o",              label="FL Loss"),
        plt.Line2D([0], [0], color=C_BL_LOSS, lw=2, marker="^", ls="--",    label="Baseline Loss"),
        plt.Line2D([0], [0], color=C_FL_ACC,  lw=2, marker="s",              label="FL Accuracy"),
        plt.Line2D([0], [0], color=C_BL_ACC,  lw=2, marker="D", ls="--",    label="Baseline Accuracy"),
    ]
    ax1.legend(handles=lines, loc="center right", frameon=True, facecolor="white", edgecolor="#b2bec3", fontsize=9)

    rounds = sorted(set(fl["round"].tolist() + bl["round"].tolist()))
    ax1.set_xticks(rounds)

    plt.title(
        "Federated Learning vs Centralized Baseline — Convergence Comparison\n"
        "(Trustworthy Federated AI · GA4GH Checkpoint-Driven Architecture)",
        fontsize=12, fontweight="bold", pad=14
    )
    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] FL vs Baseline plot → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Subpopulation size vs classwise accuracy
# ══════════════════════════════════════════════════════════════════════════════
def plot_size_vs_accuracy(sizes_csv: str, client_metrics_glob: str, out_path: str):
    if not os.path.exists(sizes_csv):
        print(f"[-] Site sizes CSV not found: {sizes_csv}")
        return

    # --- Load sizes, deduplicate by keeping last entry per site ---
    # client.py appends a row every run, so duplicates are expected
    sizes_df = pd.read_csv(sizes_csv)
    sizes_df = sizes_df.groupby("site", as_index=False).last()

    # --- Load last-round classwise accuracy from each client CSV ---
    client_files = sorted(glob.glob(client_metrics_glob))
    if not client_files:
        print(f"[-] No client metrics found matching: {client_metrics_glob}")
        return

    acc_rows = []
    for fp in client_files:
        df = pd.read_csv(fp)
        if df.empty:
            continue
        last      = df[df["round"] == df["round"].min()].iloc[-1]
        site_name = os.path.basename(fp).replace("fl_client_", "").replace("_metrics.csv", "")
        row       = {"site": site_name}
        for p in SUPERPOPS:
            row[f"acc_{p}"] = float(last.get(f"acc_{p}", 0.0))
            row[f"n_{p}"]   = int(last.get(f"n_{p}",   0))
        acc_rows.append(row)

    if not acc_rows:
        print("[-] No usable client accuracy data.")
        return

    acc_df = pd.DataFrame(acc_rows).set_index("site")

    # Align to sites present in both dataframes
    sizes_df = sizes_df.set_index("site")
    common_sites = [s for s in sizes_df.index if s in acc_df.index]
    if not common_sites:
        common_sites = list(acc_df.index)
        sizes_df = sizes_df.reindex(common_sites)

    n_sites     = len(common_sites)
    hmap_counts = np.zeros((n_sites, len(SUPERPOPS)), dtype=float)
    hmap_accs   = np.zeros((n_sites, len(SUPERPOPS)), dtype=float)

    for i, site in enumerate(common_sites):
        for j, pop in enumerate(SUPERPOPS):
            # sizes_df may store counts under the bare pop name or "n_<pop>"
            for key in (pop, f"n_{pop}"):
                if site in sizes_df.index and key in sizes_df.columns:
                    val = sizes_df.loc[site, key]
                    hmap_counts[i, j] = float(val) if np.isscalar(val) else float(val.iloc[0])
                    break
            if site in acc_df.index:
                hmap_accs[i, j] = float(acc_df.loc[site, f"acc_{pop}"])

    # Normalize per row for heatmap color
    row_totals          = hmap_counts.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    hmap_frac           = hmap_counts / row_totals

    fig = plt.figure(figsize=(16, max(5, n_sites * 1.4 + 3)))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1.2], wspace=0.35)

    # ── Panel A: Heatmap ──────────────────────────────────────────────────
    ax_heat = fig.add_subplot(gs[0])
    cmap    = LinearSegmentedColormap.from_list("fl_blue", HMAP_COLORS, N=256)
    im      = ax_heat.imshow(hmap_frac, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax_heat.set_xticks(range(len(SUPERPOPS)))
    ax_heat.set_xticklabels(SUPERPOPS, fontsize=10, fontweight="bold")
    ax_heat.set_yticks(range(n_sites))
    ax_heat.set_yticklabels(common_sites, fontsize=9)
    ax_heat.set_xlabel("Superpopulation", fontsize=10, fontweight="bold")
    ax_heat.set_ylabel("Site", fontsize=10, fontweight="bold")
    ax_heat.set_title(
        "Sample Fraction (color) & Classwise Accuracy % (text)\nper Site × Superpopulation",
        fontsize=10, fontweight="bold", pad=10
    )

    for i in range(n_sites):
        for j in range(len(SUPERPOPS)):
            txt_color = "white" if hmap_frac[i, j] > 0.45 else "#2d3436"
            ax_heat.text(j, i,
                         f"{hmap_accs[i,j]*100:.0f}%\n(n={int(hmap_counts[i,j])})",
                         ha="center", va="center", fontsize=8,
                         color=txt_color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.03)
    cbar.set_label("Fraction of site samples", fontsize=8)

    # ── Panel B: Scatter – sample count vs accuracy ───────────────────────
    ax_scat    = fig.add_subplot(gs[1])
    pop_colors = plt.cm.tab10(np.linspace(0, 0.5, len(SUPERPOPS)))

    for j, (pop, col) in enumerate(zip(SUPERPOPS, pop_colors)):
        xs = [hmap_counts[i, j] for i in range(n_sites)]
        ys = [hmap_accs[i, j] * 100 for i in range(n_sites)]
        ax_scat.scatter(xs, ys, color=col, label=pop,
                        s=80, edgecolors="white", linewidths=0.8, zorder=3)
        if len(xs) >= 2 and np.std(xs) > 0:
            z  = np.polyfit(xs, ys, 1)
            xr = np.linspace(min(xs), max(xs), 50)
            ax_scat.plot(xr, np.polyval(z, xr), color=col, lw=1.2, alpha=0.5)

    ax_scat.set_xlabel("# Samples of Superpopulation at Site", fontsize=10, fontweight="bold")
    ax_scat.set_ylabel("Classwise Accuracy (%)", fontsize=10, fontweight="bold")
    ax_scat.set_title("Does Sample Size Drive\nClasswise Accuracy?",
                      fontsize=10, fontweight="bold", pad=10)
    ax_scat.legend(fontsize=8, frameon=True, facecolor="white")
    ax_scat.grid(True, linestyle="--", alpha=0.4, color=C_GRID)
    ax_scat.set_ylim(-5, 108)

    fig.suptitle(
        "Subpopulation Representation vs Classification Performance · FL Final Round",
        fontsize=12, fontweight="bold", y=1.01
    )
    fig.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Size vs accuracy plot → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR,     exist_ok=True)

    generate_performance_plots(
        f"{CHECKPOINTS_DIR}/server_metrics.csv",
        f"{CHECKPOINTS_DIR}/federated_convergence_plot.png"
    )

    plot_fl_vs_baseline(
        fl_csv       = f"{CHECKPOINTS_DIR}/server_metrics.csv",
        baseline_csv = f"{CHECKPOINTS_DIR}/baseline_metrics.csv",
        out_path     = f"{CHECKPOINTS_DIR}/fl_vs_baseline_plot.png"
    )

    plot_size_vs_accuracy(
        sizes_csv           = f"{RESULTS_DIR}/fl_site_sizes.csv",
        client_metrics_glob = f"{RESULTS_DIR}/fl_client_*_metrics.csv",
        out_path            = f"{CHECKPOINTS_DIR}/subpop_size_vs_accuracy.png"
    )