# plot_results.py
import os
import pandas as pd
import matplotlib.pyplot as plt

def generate_performance_plots(metrics_csv_path: str, output_image_path: str):
    if not os.path.exists(metrics_csv_path):
        print(f"[-] Execution Error: Server metrics file not found at {metrics_csv_path}")
        print("    Please ensure 'python run_simulation.py' has completed successfully.")
        return

    # Ingest the server execution training history metrics log
    df = pd.read_csv(metrics_csv_path)
    
    if df.empty:
        print(f"[-] Metrics sheet {metrics_csv_path} is currently empty.")
        return

    # Setup dual-axis plotting container frame
    fig, ax1 = plt.subplots(figsize=(11, 6))

    # --- Plot Line Axis 1: Weighted Cross-Entropy Loss Reduction Curve ---
    color = '#d63031' # Production Crimson
    ax1.set_xlabel('Global Federated Rounds', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('Global Weighted Loss', color=color, fontsize=12, fontweight='bold')
    loss_line = ax1.plot(df['round'], df['weighted_loss'], color=color, marker='o', 
                         markersize=6, linewidth=2, linestyle='-', label='Federated Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- Plot Line Axis 2: Twin Weighted Superpopulation Accuracy (%) ---
    ax2 = ax1.twinx()  
    color = '#0984e3' # Production Cobalt Blue
    ax2.set_ylabel('Global Weighted Accuracy (%)', color=color, fontsize=12, fontweight='bold')
    
    # Scale fractional values to percentages if logged raw as float boundaries
    accuracy_values = df['weighted_accuracy'] if df['weighted_accuracy'].max() > 1.0 else df['weighted_accuracy'] * 100
    
    acc_line = ax2.plot(df['round'], accuracy_values, color=color, marker='s', 
                        markersize=6, linewidth=2, linestyle='-', label='Federated Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Combine plot legends across distinct axis lines natively
    lines = loss_line + acc_line
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', edgecolor='#b2bec3')

    # Format axis intervals based on round boundaries
    ax1.set_xticks(df['round'])
    
    plt.title('Trustworthy Federated AI: Genetic Ancestry Convergence Analytics\n(GA4GH Checkpoint-Driven Stateless Architecture Baseline)', 
              fontsize=13, fontweight='bold', pad=15)
    
    fig.tight_layout()
    
    # Commit visual asset to disk target
    plt.savefig(output_image_path, dpi=300)
    print(f"[+] Performance analysis graph successfully generated: {output_image_path}")

if __name__ == "__main__":
    # Ensure folder boundaries are present before writing asset
    os.makedirs("./checkpoints", exist_ok=True)
    generate_performance_plots("./checkpoints/server_metrics.csv", "./checkpoints/federated_convergence_plot.png")