#!/usr/bin/env bash

set -euo pipefail

echo "======================================================================"
echo "      GA4GH Trustworthy Federated AI - Kubernetes Pipeline"
echo "======================================================================"

ROOT_DIR="$(pwd)"

check_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Error: '$1' is not installed."
        exit 1
    }
}

echo "[1/9] Checking prerequisites..."

check_cmd docker
check_cmd kubectl
check_cmd kind
check_cmd python
check_cmd git
check_cmd plink2

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

docker build \
    -t trustworthy-fed-ai:v1 \
    -f src/Dockerfile \
    src/

docker build \
    -t funnel:local \
    -f ../funnel/Dockerfile \
    ../funnel

echo "[5/9] Preparing dataset..."

python scripts/get_flan_data.py
python scripts/prepare_splits_after_flan.py

echo "[6/9] Creating Funnel directories..."

sudo mkdir -p \
    /opt/funnel/central/work-dir /opt/funnel/central/db \
    /opt/funnel/site-1/work-dir /opt/funnel/site-1/db \
    /opt/funnel/site-2/work-dir /opt/funnel/site-2/db \
    /opt/funnel/site-3/work-dir /opt/funnel/site-3/db \
    /opt/funnel/site-4/work-dir /opt/funnel/site-4/db

sudo chmod -R 777 /opt/funnel

echo "[7/9] Creating Kind cluster..."

if kind get clusters | grep -q "^fl-cluster$"; then
    kind delete cluster --name fl-cluster
fi

kind create cluster \
    --name fl-cluster \
    --config deploy/kind/00-cluster.yaml

kind load docker-image trustworthy-fed-ai:v1 --name fl-cluster
kind load docker-image funnel:local --name fl-cluster

echo "[8/9] Deploying Kubernetes resources..."

kubectl apply \
    -f deploy/kind/01-rbac.yaml \
    -f deploy/kind/02-funnel-config.yaml \
    -f deploy/kind/03-minio.yaml \
    -f deploy/kind/04-site-central.yaml \
    -f deploy/kind/05-client-sites.yaml \
    -f deploy/kind/06-seeder-job.yaml

echo "Waiting for deployments..."

kubectl rollout status deployment/minio -n fl-system
kubectl rollout status deployment/tes-central -n fl-system

kubectl rollout status deployment/tes-site-1 -n fl-system
kubectl rollout status deployment/tes-site-2 -n fl-system
kubectl rollout status deployment/tes-site-3 -n fl-system
kubectl rollout status deployment/tes-site-4 -n fl-system

kubectl rollout status deployment/drs-central -n fl-system

kubectl rollout status deployment/drs-site-1 -n fl-system
kubectl rollout status deployment/drs-site-2 -n fl-system
kubectl rollout status deployment/drs-site-3 -n fl-system
kubectl rollout status deployment/drs-site-4 -n fl-system

echo "Waiting for DRS Seeder Job..."

kubectl wait \
    --for=condition=complete \
    job/drs-seeder-job \
    -n fl-system \
    --timeout=300s

echo "Initializing MinIO..."

kubectl exec -n fl-system deployment/minio -- \
    mc alias set myminio http://localhost:9000 minioadmin minioadmin

kubectl exec -n fl-system deployment/minio -- \
    mc mb myminio/fl-checkpoints || true

echo "[9/9] Starting orchestrator..."

python run_kind_orch.py

echo
echo "Pipeline completed."
```
