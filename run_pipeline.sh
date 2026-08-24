#!/usr/bin/env bash

set -euo pipefail

echo "======================================================================"
echo "      GA4GH Trustworthy Federated AI - Kubernetes Pipeline"
echo "======================================================================"

ROOT_DIR="$(pwd)"

check_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Error: '$1' is not installed. Please install it and try again."
        exit 1
    }
}

echo "[1/9] Checking prerequisites..."
check_cmd docker
check_cmd kubectl
check_cmd kind
check_cmd python
check_cmd git

echo "[2/9] Cloning dependencies..."
if [ ! -d "../flan" ]; then
    git clone https://github.com/TheVidz/flan.git ../flan
fi

if [ ! -d "../funnel" ]; then
    git clone https://github.com/calypr/funnel.git ../funnel
fi

echo "[3/9] Installing FLAN..."
pushd ../flan >/dev/null
pip install -e .
popd >/dev/null

echo "[4/9] Building Docker images..."
# Build the federated learning image
docker build \
    -t trustworthy-fed-ai:v1 \
    -f src/Dockerfile \
    src/

# Build the Funnel image using the custom Dockerfile.funnel
docker build \
    -t funnel:local \
    -f Dockerfile.funnel \
    ../funnel

echo "[5/9] Downloading and preprocessing the dataset..."
python scripts/get_flan_data.py
python scripts/prepare_splits_after_flan.py

echo "[6/9] Creating Kind cluster..."
if kind get clusters | grep -q "^fl-cluster$"; then
    echo "Cluster 'fl-cluster' already exists. Deleting it to start fresh..."
    kind delete cluster --name fl-cluster
fi

kind create cluster \
    --name fl-cluster \
    --config deploy/kind/00-cluster.yaml

echo "[7/9] Loading Docker images into Kind..."
kind load docker-image trustworthy-fed-ai:v1 --name fl-cluster
kind load docker-image funnel:local --name fl-cluster

echo "[8/9] Deploying Kubernetes infrastructure..."
kubectl apply \
    -f deploy/kind/01-rbac.yaml \
    -f deploy/kind/02-funnel-config.yaml \
    -f deploy/kind/03-minio.yaml \
    -f deploy/kind/04-site-central.yaml \
    -f deploy/kind/05-client-sites.yaml \
    -f deploy/kind/06-seeder-job.yaml

echo "Waiting for deployments to roll out..."
kubectl rollout status deployment/minio -n fl-system
kubectl rollout status deployment/tes-central -n fl-system

for i in {1..4}; do
    kubectl rollout status deployment/tes-site-$i -n fl-system
    kubectl rollout status deployment/drs-site-$i -n fl-system
done

kubectl rollout status deployment/drs-central -n fl-system

echo "Waiting for DRS Seeder Job to complete..."
kubectl wait \
    --for=condition=complete \
    job/drs-seeder-job \
    -n fl-system \
    --timeout=300s

echo "Initializing and configuring MinIO..."
# Wait a few seconds to ensure MinIO is fully ready to accept connections
sleep 5 
kubectl exec -n fl-system deployment/minio -- \
    mc alias set myminio http://localhost:9000 minioadmin minioadmin

kubectl exec -n fl-system deployment/minio -- \
    mc mb myminio/fl-checkpoints || true

echo "[9/9] Starting federated learning orchestrator..."
python run_orch.py

echo "======================================================================"
echo "Pipeline completed successfully."
echo "You can view convergence metrics by running: python plot_results.py"
echo "======================================================================"
