# run_simulation.py
import os
import subprocess
import sys

ROUNDS = 15
NUM_CLIENTS = 4
ARTIFACTS_DIR = "./checkpoints"

def run_cmd(cmd):
    """Executes a standard batch container shell script command synchronously."""
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"\n[Execution Failure] Command failed with exit code: {process.returncode}")
        sys.exit(process.returncode)

def main():
    print("=========================================================================")
    print("   Starting Automated Trusted Federated AI Production-Grade Sandbox Flow")
    print("=========================================================================")
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Step 1: Bootstrap baseline global matrix model checkpoint (Round 0 initialization)
    print("\n--- Phase 0: Launching Server Initialization ---")
    bootstrap_cmd = f"python server.py --target-round 0 --num-clients {NUM_CLIENTS} --artifacts-dir {ARTIFACTS_DIR}"
    run_cmd(bootstrap_cmd)

    current_global_weights = os.path.join(ARTIFACTS_DIR, "global_model_round_0.pt")

    # Step 2: Main Automated Iteration Loop Engine
    for r in range(1, ROUNDS + 1):
        print(f"\n\n#################################################################")
        print(f"   EXECUTING FEDERATED ROUND {r} / {ROUNDS}")
        print(f"#################################################################")

        # Sequentially trigger each ephemeral worker task payload
        for c_id in range(1, NUM_CLIENTS + 1):
            client_out = os.path.join(ARTIFACTS_DIR, f"client_{c_id}_round_{r}.pt")
            unified_id = f"site_{c_id}_unified"
            
            print(f"\n[Automator Launcher] Spawning Worker Node Payload for Client {c_id}")
            # FIXED: Argument flag changed from --drs-id to --unified-id to match client.py
            client_cmd = (
                f"python client.py --client-id {c_id} --unified-id {unified_id} "
                f"--global-weights-path {current_global_weights} --output-weights-path {client_out} "
                f"--epochs 5 --batch-size 16 --lr 0.01"
            )
            run_cmd(client_cmd)

        # Trigger Server Aggregator to read client updates and merge them
        print(f"\n[Automator Aggregator] Invoking Master FedAvg Core Layer for Round {r}")
        server_cmd = f"python server.py --target-round {r} --num-clients {NUM_CLIENTS} --artifacts-dir {ARTIFACTS_DIR}"
        run_cmd(server_cmd)

        # Update pointers for the next epoch round sequence
        current_global_weights = os.path.join(ARTIFACTS_DIR, f"global_model_round_{r}.pt")

    print("\n=========================================================================")
    print("   Federated Iteration Loop Complete! Launching Plot Evaluation Curves.")
    print("=========================================================================")
    run_cmd("python plot_results.py")

if __name__ == "__main__":
    main()