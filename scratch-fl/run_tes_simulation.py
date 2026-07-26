# run_simulation.py
import os
import sys
import time
import tes  # Core GA4GH TES Python SDK Client Wrapper
import threading

ROUNDS = 5
NUM_CLIENTS = 4
IMAGE_TAG = "trustworthy-fed-ai:v1"

HOST_SHARED_DIR = os.path.abspath("./tmp/tes-workspace")
os.makedirs(HOST_SHARED_DIR, exist_ok=True)

# Connect the py-tes client directly to Funnel socket port
client = tes.HTTPClient("http://localhost:8000")

def execute_and_wait(task_object):
    """Submits structural task definitions to Funnel server and polls execution loop bounds."""
    task_id = client.create_task(task_object)
    print(f"[py-tes Broker] Task dispatched safely. Assigned tracking ID: {task_id}")
    
    while True:
        status = client.get_task(task_id, view="MINIMAL")
        state = status.state
        
        if state == "COMPLETE":
            print(f"[✓] Task {task_id} successfully finalized processing boundaries.")
            break
        elif state in ["EXECUTOR_ERROR", "SYSTEM_ERROR", "CANCELED"]:
            print(f"[-] Critical Crash Alert: Task {task_id} collapsed with state: {state}")
            full_log = client.get_task(task_id, view="FULL")
            for log in full_log.logs:
                for ex_log in log.logs:
                    print(f"Container Stack Trace Log Dump:\n{ex_log.stderr}")
os._exit(1)
            
        print(f"    [Polling] Task State: {state}... re-verifying in 3 seconds.")
        time.sleep(3)

def main():
    print("=========================================================================")
    print("   Igniting Source-Compiled GA4GH TES Checkpoint Federated Pipeline      ")
    print("=========================================================================")

    # --- Step 0: Bootstrap Initial Global Parameters Weights ---
    print("\n[Phase 0] Launching Server Initialization Container...")
    round_0_task = tes.Task(
        name="Server-Weight-Bootstrap-Round-0",
        executors=[
            tes.Executor(
                image=IMAGE_TAG,
                command=["python", "server.py", "--target-round", "0", "--num-clients", str(NUM_CLIENTS), "--artifacts-dir", "/workspace/checkpoints", "--metrics-path", "/workspace/checkpoints/server_metrics.csv"]
            )
        ],
        outputs=[
            tes.Output(
                path="/workspace/checkpoints/global_model_round_0.pt",
                url=f"file://{HOST_SHARED_DIR}/global_model_round_0.pt",
                type="FILE"
            )
        ]
    )
    execute_and_wait(round_0_task)

    # --- Multi-Round Federated Averaging State Machine ---
    for r in range(1, ROUNDS + 1):
        print(f"\n\n#################################################################")
        print(f"   DISPATCHING COMPUTE TASKS FOR FEDERATED TRAIN ROUND {r} / {ROUNDS}")
        print(f"#################################################################")

        # Step 1: Run Client Silo Nodes concurrently using Threads
        threads = []
        
        for c_id in range(1, NUM_CLIENTS + 1):
            print(f"\n[Orchestrator] Building Isolated Hospital Node Job Profile: Client {c_id}")
            client_task = tes.Task(
                name=f"Client_Site_{c_id}_Round_{r}",
                inputs=[
                    tes.Input(
                        path="/workspace/checkpoints/global_model_current.pt",
                        url=f"file://{HOST_SHARED_DIR}/global_model_round_{r-1}.pt",
                        type="FILE"
                    )
                ],
                executors=[
                    tes.Executor(
                        image=IMAGE_TAG,
                        command=[
                            "python", "client.py",
                            "--client-id", str(c_id),
                            "--unified-id", f"site_{c_id}_unified",
                            "--global-weights-path", "/workspace/checkpoints/global_model_current.pt",
                            "--output-weights-path", f"/workspace/checkpoints/client_{c_id}_round_{r}.pt",
                            "--results-dir", "/workspace/checkpoints",
                            "--epochs", "5"
                        ]
                    )
                ],
                outputs=[
                    tes.Output(
                        path=f"/workspace/checkpoints/client_{c_id}_round_{r}.pt",
                        url=f"file://{HOST_SHARED_DIR}/client_{c_id}_round_{r}.pt",
                        type="FILE"
                    )
                ]
            )
            
            # Create a thread for each client task so they submit and poll concurrently
            t = threading.Thread(target=execute_and_wait, args=(client_task,))
            threads.append(t)
            t.start() # Starts execution asynchronously

        # Block the main orchestrator loop until ALL client threads have completed
        for t in threads:
            t.join()

        # Step 2: Trigger Central Weight Averaging Task
        print(f"\n[Orchestrator] Gathering check-ins to spin up Central FedAvg Aggregator")
        server_inputs = [
            tes.Input(
                path=f"/workspace/checkpoints/client_{c}_round_{r}.pt",
                url=f"file://{HOST_SHARED_DIR}/client_{c}_round_{r}.pt",
                type="FILE"
            ) for c in range(1, NUM_CLIENTS + 1)
        ]

        server_task = tes.Task(
            name=f"Server_Mathematical_Aggregation_Round_{r}",
            inputs=server_inputs,
            executors=[
                tes.Executor(
                    image=IMAGE_TAG,
                    command=[
                        "python", "server.py",
                        "--target-round", str(r),
                        "--num-clients", str(NUM_CLIENTS),
                        "--artifacts-dir", "/workspace/checkpoints",
                        "--metrics-path", "/workspace/checkpoints/server_metrics.csv"
                    ]
                )
            ],
            outputs=[
                tes.Output(
                    path=f"/workspace/checkpoints/global_model_round_{r}.pt",
                    url=f"file://{HOST_SHARED_DIR}/global_model_round_{r}.pt",
                    type="FILE"
                ),
                tes.Output(
                    path="/workspace/checkpoints/server_metrics.csv",
                    url=f"file://{HOST_SHARED_DIR}/server_metrics.csv",
                    type="FILE"
                )
            ]
        )
        execute_and_wait(server_task)

    print("\n[✓] All optimization lifecycles safely parsed without connection decay errors.")

if __name__ == "__main__":
    main()