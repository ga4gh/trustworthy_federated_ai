#  GA4GH Trustworthy Federated AI: Genomic Ancestry Prediction

[![Standards](https://img.shields.io/badge/GA4GH-TES%20%7C%20DRS-blue)](https://www.ga4gh.org/)
[![Orchestration](https://img.shields.io/badge/Kubernetes-Kind-green)](https://kind.sigs.k8s.io/)
[![Storage](https://img.shields.io/badge/S3-MinIO-red)](https://min.io/)
[![Dataset](https://img.shields.io/badge/Dataset-1000%20Genomes-orange)](https://www.internationalgenome.org/)

A proof-of-concept for federated learning on genomic data using GA4GH standards. The project demonstrates distributed ancestry classification across multiple simulated institutions using DRS, TES, Kubernetes, and MinIO.


## 🛠️ Components & Tooling Versions

The architecture integrates standardized bioinformatics pipelines with cloud-native distributed components:

| Component | Role in Architecture | Version | Documentation / Source |
| :--- | :--- | :--- | :--- |
| **[PLINK2](https://www.cog-genomics.org/plink/2.0/)** | High-performance bioinformatics engine for Variant Quality Control (QC) and Principal Component Analysis (PCA). | `v2.00a4LM` | [PLINK2 Toolset](https://www.cog-genomics.org/plink/2.0/) |
| **[GA4GH DRS](https://github.com/ga4gh/ga4gh-starter-kit-drs)** | Data Repository Service providing storage-agnostic, secure URI-based access (`drs://`) to local or cloud genetic files. | `v0.3.2` | [GA4GH DRS API](https://github.com/ga4gh/ga4gh-starter-kit-drs) |
| **[GA4GH TES (Funnel)](https://github.com/calypr/funnel)** | Task Execution Service acting as the backend engine to schedule, isolate, and execute ephemeral training containers. | `v0.12.0` | [GA4GH TES / Funnel](https://github.com/calypr/funnel) |
| **[Kind (Kubernetes IN Docker)](https://kind.sigs.k8s.io/)** | Local multi-node Kubernetes cluster provider used to simulate multi-institution network firewalls. | `v0.20.0+` | [Kind Documentation](https://kind.sigs.k8s.io/) |
| **[MinIO](https://min.io/)** | S3-compatible object storage layer used for decentralized model parameter checkpoint exchanges (`.pt`). | `RELEASE.2023-*` | [MinIO Storage](https://min.io/) |
| **[PyTorch](https://pytorch.org/)** | Deep learning framework executing local Multi-Layer Perceptron (MLP) training and FedAvg parameter calculations. | `v2.1.0+CPU` | [PyTorch Framework](https://pytorch.org/) |
| **[py-tes](https://github.com/ohsu-comp-bio/py-tes)** | Python SDK for interacting with GA4GH Task Execution Service (TES) endpoints. | `v0.4.2+` | [py-tes Client](https://github.com/ohsu-comp-bio/py-tes) |

---

## 🔬 Scientific Process & Data Lineage

Rather than passing massive, multi-gigabyte raw VCF (Variant Call Format) files over network sockets—which causes severe memory overhead and privacy breaches—the pipeline follows a strict privacy-preserving bioinformatics protocol:

1. **Feature Compression via PCA (PLINK2):** Raw variant arrays from the 1000 Genomes Project (~900,000 SNPs) are processed locally. PLINK2 executes Variant Quality Control (filtering for Minor Allele Frequency `--maf 0.15`, missingness `--geno 0.02`, and Hardy-Weinberg equilibrium `--hwe 0.000001`) and computes the top 10 Principal Components (`.sscore`).
2. **Tabular Coordinate Mapping:** The deep learning model completely skips raw DNA bases, operating instead on a lightweight 10-dimensional coordinate vector representing the patient's position in global genetic space.
3. **Superpopulation Targets:** The classification target maps precisely to 5 global superpopulations defined in international genomics studies: **AFR** (African), **AMR** (Admixed American), **EAS** (East Asian), **EUR** (European), and **SAS** (South Asian).
4. **Stateless Federated Averaging (FedAvg):** Instead of maintaining open streaming sockets (like legacy gRPC frameworks), training operates as a **Stateless Directed Acyclic Graph (DAG)**:
   * The Central Aggregator initializes a global model checkpoint (`global_model_round_0.pt`) in MinIO object storage.
   * Client sites spin up ephemeral containers via TES, pull global weights, train locally for $E$ epochs on private DRS-resolved data splits, and upload updated parameter weights (`.pt`) back to MinIO.
   * The Central Server aggregates parameters using weighted FedAvg without ever viewing individual patient records.

---

## 📦 Component Execution Roles

* **`server.py` (Central Aggregator):** Manages global training rounds, pulls client checkpoints from MinIO, executes mathematical weight averaging (FedAvg), and evaluates global test metrics.
* **`client.py` (Edge Worker):** An ephemeral containerized script that resolves its training and validation splits via DRS, loads the 10-feature coordinate tensors into the `AncestryNet` MLP, trains locally, and outputs updated weights.
* **`run_kind_orch.py` (The Orchestrator):** Dispatches standard JSON task payloads (`tes.Task`) to individual institutional TES endpoints via multi-threading, acting as the workflow conductor.
* **GA4GH DRS (Starter Kit):** Serves metadata and access URIs (`drs://`) for institutional datasets, decoupling storage locations from computation.
* **GA4GH TES (Funnel):** Accepts task submissions over REST API and manages container execution environments across Kubernetes pods.
* **MinIO Storage:** S3-compatible object store hosting global model checkpoints and client weight updates in bucket `s3://fl-checkpoints/`.

---

## 🗺️ System Architecture & Kubernetes Mapping

The architecture models multi-institution firewalls by deploying isolated namespaces, local task engines, and object storage within a single Kind cluster (`fl-cluster`) under the `fl-system` namespace:

# insert diagram

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

### 2. Create the Kind cluster

```bash
kind create cluster --config deploy/kind/00-cluster.yaml
```

### 3. Load Docker images

```bash
kind load docker-image trustworthy-fed-ai:v1 --name fl-cluster

kind load docker-image funnel:local --name fl-cluster
```

### 4. Deploy the infrastructure

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

### 5. Configure MinIO

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
