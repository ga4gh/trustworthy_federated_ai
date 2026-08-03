import os
import sys
import time
import tes
import threading

ROUNDS = 5
SITES = ['1', '2', '3', '4']
IMAGE_TAG = "trustworthy-fed-ai:v1"
BUCKET = "fl-checkpoints"

# Host-accessible NodePort mappings for TES clients
TES_CLIENTS = {
    'central': tes.HTTPClient("http://localhost:8000"),
    '1': tes.HTTPClient("http://localhost:8001"),
    '2': tes.HTTPClient("http://localhost:8002"),
    '3': tes.HTTPClient("http://localhost:8003"),
    '4': tes.HTTPClient("http://localhost:8004")
}

# DRS endpoints using host gateway IP
HOST_GATEWAY = "172.17.0.1"
CENTRAL_DRS_ENDPOINT = f"http://{HOST_GATEWAY}:4500"
DRS_ENDPOINTS = {
    "central": f"http://{HOST_GATEWAY}:4500",
    "1": f"http://{HOST_GATEWAY}:4502",
    "2": f"http://{HOST_GATEWAY}:4504",
    "3": f"http://{HOST_GATEWAY}:4506",
    "4": f"http://{HOST_GATEWAY}:4508",
}

def execute_and_wait(task_object, tes_client, node_name="Unknown"):
    """Submits tasks to a specific Funnel endpoint and polls for completion."""
    task_id = tes_client.create_task(task_object)
    print(f"[Orchestrator] Task dispatched to {node_name}. ID: {task_id}")
    
    while True:
        status = tes_client.get_task(task_id, view="MINIMAL")
        state = status.state
        
        if state == "COMPLETE":
            print(f"[✓] Task {task_id} on {node_name} completed successfully.")
            break
        elif state in ["EXECUTOR_ERROR", "SYSTEM_ERROR", "CANCELED"]:
            print(f"[-] Error: Task {task_id} on {node_name} failed with state: {state}")
            sys.exit(1)
            
        time.sleep(3)

def main():
    print("=========================================================================")
    print("   Igniting Multi-DRS Federated Learning Pipeline (Kubernetes Native)   ")
    print("=========================================================================")

    # --- Phase 0: Bootstrap Initial Global Weights ---
    print("\n[Phase 0] Initializing Global Model on Central Server...")
    round_0_task = tes.Task(
        name="Server-Bootstrap",
        executors=[
            tes.Executor(
                image=IMAGE_TAG,
                command=[
                    "python", "server.py", 
                    "--target-round", "0", 
                    "--drs-endpoint", CENTRAL_DRS_ENDPOINT,
                    "--artifacts-dir", "/workspace/checkpoints", 
                    "--metrics-path", "/workspace/checkpoints/server_metrics_round_0.csv"
                ]
            )
        ],
        outputs=[
            tes.Output(
                path="/workspace/checkpoints/global_model_round_0.pt",
                url=f"s3://{BUCKET}/models/global_model_round_0.pt",
                type="FILE"
            ),
            tes.Output(
                path="/workspace/checkpoints/server_metrics_round_0.csv",
                url=f"s3://{BUCKET}/metrics/server_metrics_round_0.csv",
                type="FILE"
            )
        ]
    )
    execute_and_wait(round_0_task, TES_CLIENTS['central'], "Central-Node")

    # --- Multi-Round FL State Machine ---
    for r in range(1, ROUNDS + 1):
        print(f"\n#################################################################")
        print(f"   DISPATCHING COMPUTE TASKS FOR ROUND {r} / {ROUNDS}")
        print(f"#################################################################")

        threads = []
        for site in SITES:
            client_task = tes.Task(
                name=f"Client_Site_{site}_Round_{r}",
                inputs=[
                    tes.Input(
                        url=f"s3://{BUCKET}/models/global_model_round_{r-1}.pt",
                        path=f"/workspace/checkpoints/global_model_round_{r-1}.pt",
                        type="FILE"
                    )
                ],
                executors=[
                    tes.Executor(
                        image=IMAGE_TAG,
                        command=[
                            "python", "client.py",
                            "--site-id", site,
                            "--drs-endpoint", DRS_ENDPOINTS[site],
                            "--global-weights-path", f"/workspace/checkpoints/global_model_round_{r-1}.pt",
                            "--output-weights-path", f"/workspace/checkpoints/client_{site}_round_{r}.pt",
                            "--results-dir", "/workspace/checkpoints",
                            "--epochs", "5"
                        ]
                    )
                ],
                outputs=[
                    tes.Output(
                        url=f"s3://{BUCKET}/clients/client_{site}_round_{r}.pt",
                        path=f"/workspace/checkpoints/client_{site}_round_{r}.pt",
                        type="FILE"
                    ),
                    tes.Output(
                        path=f"/workspace/checkpoints/fl_client_site_{site}_metrics.csv",
                        url=f"s3://{BUCKET}/metrics/fl_client_site_{site}_round_{r}_metrics.csv",
                        type="FILE"
                    )
                ]
            )
            site_client = TES_CLIENTS[site]
            t = threading.Thread(target=execute_and_wait, args=(client_task, site_client, f"Site-{site}"))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Step 2: Central Aggregation & Global Evaluation
        print(f"\n[Orchestrator] Triggering Central Aggregation and Global Test...")
        server_inputs = [
            tes.Input(
                path=f"/workspace/checkpoints/client_{site}_round_{r}.pt",
                url=f"s3://{BUCKET}/clients/client_{site}_round_{r}.pt",
                type="FILE"
            ) for site in SITES
        ]

        server_task = tes.Task(
            name=f"Server_Aggregation_Round_{r}",
            inputs=server_inputs,
            executors=[
                tes.Executor(
                    image=IMAGE_TAG,
                    command=[
                        "python", "server.py",
                        "--target-round", str(r),
                        "--drs-endpoint", CENTRAL_DRS_ENDPOINT,
                        "--sites", ",".join(SITES),
                        "--artifacts-dir", "/workspace/checkpoints",
                        "--metrics-path", f"/workspace/checkpoints/server_metrics_round_{r}.csv"
                    ]
                )
            ],
            outputs=[
                tes.Output(
                    path=f"/workspace/checkpoints/global_model_round_{r}.pt",
                    url=f"s3://{BUCKET}/models/global_model_round_{r}.pt",
                    type="FILE"
                ),
                tes.Output(
                    path=f"/workspace/checkpoints/server_metrics_round_{r}.csv",
                    url=f"s3://{BUCKET}/metrics/server_metrics_round_{r}.csv",
                    type="FILE"
                )
            ]
        )
        execute_and_wait(server_task, TES_CLIENTS['central'], "Central-Node")

    print("\n[✓] Kubernetes Federated Pipeline completed successfully.")

if __name__ == "__main__":
    main()