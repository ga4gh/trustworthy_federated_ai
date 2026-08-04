# GA4GH Trustworthy Federated AI in Genomics

[![Standards](https://img.shields.io/badge/GA4GH-TES%20%7C%20DRS-blue)](https://www.ga4gh.org/)
[![Orchestration](https://img.shields.io/badge/Kubernetes-Kind-green)](https://kind.sigs.k8s.io/)
[![Storage](https://img.shields.io/badge/S3-MinIO-red)](https://min.io/)
[![Dataset](https://img.shields.io/badge/Dataset-1000%20Genomes-orange)](https://www.internationalgenome.org/)

A proof-of-concept for federated learning on genomic data using GA4GH standards. The project demonstrates distributed ancestry classification across multiple simulated institutions using DRS, TES, Kubernetes, and MinIO.

## Components

* GA4GH Data Repository Service (DRS)
* GA4GH Task Execution Service (TES/Funnel)
* Kind Kubernetes cluster
* MinIO (S3-compatible object storage)
* PLINK2 preprocessing pipeline
* PyTorch Federated Averaging

---

## Prerequisites

Install:

* Docker
* Kind
* kubectl
* Git
* Conda / Miniconda

---

## Setup

Clone the required repositories:

```bash
git clone https://github.com/ga4gh/trustworthy_federated_ai.git
git clone https://github.com/TheVidz/flan.git
git clone https://github.com/calypr/funnel.git
```

Create the environment:

```bash
cd trustworthy_federated_ai

conda env create -f environment.yml
conda activate flan-env
```

Install FLAN (required for the `plink2` wrapper used during preprocessing):

```bash
cd ../flan
pip install -e .

cd ../trustworthy_federated_ai
```

Build the Docker images:

```bash
docker build -t trustworthy-fed-ai:v1 -f src/Dockerfile src/

docker build -t funnel:local -f ../funnel/Dockerfile ../funnel
```

---

## Running

### 1. Download and preprocess the dataset

```bash
python scripts/get_flan_data.py

python scripts/prepare_splits_after_flan.py
```

### 2. Create Funnel directories

```bash
sudo mkdir -p \
    /opt/funnel/central/work-dir /opt/funnel/central/db \
    /opt/funnel/site-1/work-dir /opt/funnel/site-1/db \
    /opt/funnel/site-2/work-dir /opt/funnel/site-2/db \
    /opt/funnel/site-3/work-dir /opt/funnel/site-3/db \
    /opt/funnel/site-4/work-dir /opt/funnel/site-4/db

sudo chmod -R 777 /opt/funnel
```

### 3. Create the Kind cluster

```bash
kind create cluster --config deploy/kind/00-cluster.yaml
```

### 4. Load Docker images

```bash
kind load docker-image trustworthy-fed-ai:v1 --name fl-cluster

kind load docker-image funnel:local --name fl-cluster
```

### 5. Deploy the infrastructure

```bash
kubectl apply \
    -f deploy/kind/01-rbac.yaml \
    -f deploy/kind/02-funnel-config.yaml \
    -f deploy/kind/03-minio.yaml \
    -f deploy/kind/04-site-central.yaml \
    -f deploy/kind/05-client-sites.yaml \
    -f deploy/kind/06-seeder-job.yaml
```

Wait until all pods are running:

```bash
kubectl get pods -n fl-system -w
```

The seeder job should finish with `Completed` before continuing.

### 6. Configure MinIO

```bash
kubectl exec -n fl-system deployment/minio -- \
    mc alias set myminio http://localhost:9000 minioadmin minioadmin

kubectl exec -n fl-system deployment/minio -- \
    mc mb myminio/fl-checkpoints || true
```

### 7. Start federated learning

```bash
python run_kind_orch.py
```

---

## Repository Structure

```text
.
├── configs/
├── deploy/
│   └── kind/
├── scripts/
├── src/
├── run_kind_orch.py
├── environment.yml
└── README.md
```

---

## Model

* **Features:** Top 10 PCA components generated with PLINK2
* **Classes:** AFR, AMR, EAS, EUR, SAS
* **Model:** PyTorch MLP
* **Aggregation:** Federated Averaging (FedAvg)

---

## License

Apache License 2.0.
