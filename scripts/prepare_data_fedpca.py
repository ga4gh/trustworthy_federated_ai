import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import plotly.express as px

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# =====================================================================
# CONSTANTS & METADATA
# =====================================================================
POPULATION_TO_SUPERPOP = {
    'ACB': 'AFR', 'ASW': 'AFR', 'ESN': 'AFR', 'GWD': 'AFR', 'LWK': 'AFR', 'MSL': 'AFR', 'YRI': 'AFR',
    'CLM': 'AMR', 'MXL': 'AMR', 'PEL': 'AMR', 'PUR': 'AMR',
    'CDX': 'EAS', 'CHB': 'EAS', 'CHS': 'EAS', 'JPT': 'EAS', 'KHV': 'EAS',
    'CEU': 'EUR', 'FIN': 'EUR', 'GBR': 'EUR', 'IBS': 'EUR', 'TSI': 'EUR',
    'BEB': 'SAS', 'GIH': 'SAS', 'ITU': 'SAS', 'PJL': 'SAS', 'STU': 'SAS'
}

SUPERPOPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
POP_TO_IDX = {"AFR": 0, "AMR": 1, "EAS": 2, "EUR": 3, "SAS": 4}

AFFY_VCF_URL = "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/ALL.wgs.nhgri_coriell_affy_6.20140825.genotypes_has_ped.vcf.gz"
PANEL_URL = "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/affy_samples.20141118.panel"

# =====================================================================
# UTILITIES & PLINK RUNNER
# =====================================================================
def run_plink(args_list: List[str], args_dict: Optional[dict] = None) -> None:
    """Executes PLINK 2.0 subprocess safely."""
    kv_pairs = [[k, str(v)] for k, v in args_dict.items()] if args_dict else []
    plink_args = ['plink2'] + args_list + [x for xs in kv_pairs for x in xs]
    logging.debug(f"PLINK cmd: {' '.join(plink_args)}")
    
    result = subprocess.run(plink_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        logging.error(result.stdout.decode('utf-8'))
        raise RuntimeError(result.stderr.decode('utf-8'))

def download_file(link: str, output_path: Path) -> None:
    """Downloads a remote file with progress bar indicator."""
    if output_path.exists():
        logging.info(f"File {output_path.name} already exists. Skipping download.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(total=100, desc=f"Downloading {output_path.name}", unit="MB") as pbar:
        def reporthook(blocknum, blocksize, totalsize):
            pbar.update(blocknum * blocksize // 1e6)
        urlretrieve(link, output_path, reporthook)
    logging.info(f"Successfully downloaded to {output_path}")

# =====================================================================
# DATA MANAGEMENT & ORCHESTRATOR (OPTION 1: REFERENCE PANEL PROJECTION)
# =====================================================================
class FederatedDataOrchestrator:
    """
    Manages raw genotype downloading, QC, and non-IID partitioning.
    Uses the Central Holdout set to define the Global PCA Reference Space via Scikit-Learn,
    bypassing complex network loops and PLINK scoring errors.
    """
    def __init__(self, cache_dir: Path, out_dir: Path, num_sites: int = 4, dominant_share: float = 0.85):
        self.cache_dir = cache_dir
        self.out_dir = out_dir
        self.num_sites = num_sites
        self.dominant_share = dominant_share

        # Cache paths
        self.vcf_path = self.cache_dir / "affymetrix.vcf.gz"
        self.panel_path = self.cache_dir / "affy_samples.panel"
        self.pfile_raw = self.cache_dir / "genotypes" / "raw_genotype"
        self.pfile_qc = self.cache_dir / "genotypes" / "qc_genotype"
        self.prune_prefix = self.cache_dir / "genotypes" / "pruned_snps"

        # Setup directory tree
        for d in [self.cache_dir / "genotypes", self.cache_dir / "splits", 
                  self.out_dir / "central", self.out_dir / "checkpoints", self.out_dir / "plots"]:
            d.mkdir(parents=True, exist_ok=True)

        for i in range(1, self.num_sites + 1):
            (self.out_dir / f"site_{i}").mkdir(parents=True, exist_ok=True)

    def prepare_raw_genotypes(self) -> None:
        """Downloads Affymetrix 1000G data, applies standard QC and LD pruning."""
        logging.info("--- Step 1: Downloading Genotypes & Metadata ---")
        download_file(AFFY_VCF_URL, self.vcf_path)
        download_file(AFFY_VCF_URL + ".tbi", self.cache_dir / "affymetrix.vcf.gz.tbi")
        download_file(PANEL_URL, self.panel_path)

        logging.info("--- Step 2: Ingesting into PLINK 2.0 PGEN format ---")
        # Correctly read native header
        pheno_df = pd.read_csv(self.panel_path, sep=r'\s+', header=0)
        
        valid_samples = pheno_df[pheno_df['pop'].isin(POPULATION_TO_SUPERPOP)][['sample']]
        keep_samples_file = self.cache_dir / "keep.samples"
        valid_samples.to_csv(keep_samples_file, sep='\t', index=False, header=False)

        run_plink(
            args_list=['--make-pgen'],
            args_dict={
                '--vcf': str(self.vcf_path),
                '--keep': str(keep_samples_file),
                '--out': str(self.pfile_raw)
            }
        )

        logging.info("--- Step 3: Performing Combined QC (MAF, GENO, MIND, HWE) ---")
        run_plink(
            args_list=['--pfile', str(self.pfile_raw), '--make-pgen'],
            args_dict={
                '--geno': '0.05',
                '--mind': '0.05',
                '--maf': '0.05',
                '--hwe': '1e-6',
                '--set-missing-var-ids': '@:#:$r:$a',
                '--out': str(self.pfile_qc)
            }
        )

        logging.info("--- Step 4: LD Pruning for Uncorrelated Genetic Markers ---")
        run_plink(
            args_list=[
                '--pfile', str(self.pfile_qc),
                '--indep-pairwise', '50', '10', '0.1'
            ],
            args_dict={
                '--out': str(self.prune_prefix)
            }
        )

    def shard_and_export_sites(self) -> Dict[str, Dict[str, Path]]:
        """
        Shards samples into Global Test (Central Server) and Local Site Train/Val
        with realistic non-IID ancestry distributions, then exports raw dosages.
        """
        logging.info("--- Step 5: Partitioning Cohorts Across Dynamic Client Sites ---")
        
        # Load sample metadata
        pheno = pd.read_csv(self.panel_path, sep=r'\s+', header=0)
        pheno.rename(columns={'sample': 'IID'}, inplace=True)
        pheno = pheno[pheno['pop'].isin(POPULATION_TO_SUPERPOP)].copy()
        pheno['super_pop'] = pheno['pop'].map(POPULATION_TO_SUPERPOP)

        # 1. Global Server Reference/Holdout Set (20%)
        train_val_df, test_df = train_test_split(pheno, test_size=0.20, random_state=42, stratify=pheno['super_pop'])
        
        # 2. Assign remaining samples to client sites with non-IID demographic skew
        site_names = [f"site_{i}" for i in range(1, self.num_sites + 1)]
        site_dominant_pop = {site_names[i]: SUPERPOPS[i % len(SUPERPOPS)] for i in range(self.num_sites)}
        
        pop_to_home = {pop: [] for pop in SUPERPOPS}
        for site, pop in site_dominant_pop.items():
            pop_to_home[pop].append(site)

        rng = np.random.RandomState(42)
        assigned_sites = []
        for pop in train_val_df['super_pop']:
            home = pop_to_home.get(pop, [])
            if home and rng.rand() < self.dominant_share:
                assigned_sites.append(rng.choice(home))
            else:
                others = [s for s in site_names if s not in home]
                assigned_sites.append(rng.choice(others) if others else rng.choice(site_names))

        train_val_df = train_val_df.copy()
        train_val_df['assigned_site'] = assigned_sites

        site_paths: Dict[str, Dict[str, Path]] = {s: {} for s in site_names}
        prune_in_file = f"{self.prune_prefix}.prune.in"

        # Export Central Test Raw Genotypes
        test_ids_path = self.cache_dir / "splits" / "test_ids.txt"
        test_df[['IID']].to_csv(test_ids_path, sep='\t', index=False, header=False)
        central_test_raw = self.cache_dir / "splits" / "central_test"
        
        run_plink(
            args_list=['--pfile', str(self.pfile_qc), '--export', 'A'],
            args_dict={
                '--keep': str(test_ids_path),
                '--extract': str(prune_in_file),
                '--out': str(central_test_raw)
            }
        )
        self.central_test_raw_file = Path(str(central_test_raw) + ".raw")
        self.test_metadata = test_df[['IID', 'super_pop']]

        # Export Local Site Train and Val Raw Genotypes
        for site in site_names:
            site_df = train_val_df[train_val_df['assigned_site'] == site]
            s_train, s_val = train_test_split(site_df, test_size=0.20, random_state=42, stratify=site_df['super_pop'])

            for split_name, s_subset in [('train', s_train), ('val', s_val)]:
                id_file = self.cache_dir / "splits" / f"{site}_{split_name}_ids.txt"
                s_subset[['IID']].to_csv(id_file, sep='\t', index=False, header=False)
                
                raw_out_prefix = self.cache_dir / "splits" / f"{site}_{split_name}"
                run_plink(
                    args_list=['--pfile', str(self.pfile_qc), '--export', 'A'],
                    args_dict={
                        '--keep': str(id_file),
                        '--extract': str(prune_in_file),
                        '--out': str(raw_out_prefix)
                    }
                )
                site_paths[site][split_name] = Path(str(raw_out_prefix) + ".raw")
                
            logging.info(f"[{site}] Sliced {len(s_train)} Train | {len(s_val)} Val samples. Dominant: {site_dominant_pop[site]}")

        self.train_val_metadata = train_val_df[['IID', 'super_pop']]
        return site_paths

    def run_reference_panel_pipeline(self) -> None:
        """Executes full pipeline utilizing Scikit-Learn Reference Panel Projection."""
        self.prepare_raw_genotypes()
        site_raw_paths = self.shard_and_export_sites()

        # Save Label Mapping JSON
        with open(self.out_dir / "checkpoints" / "label_mapping.json", "w") as f:
            json.dump(POP_TO_IDX, f, indent=2)

        logging.info("--- Step 6: Deriving Global PCA Reference Space (Central Server) ---")
        
        # 1. Load Central Reference Matrix
        central_df = pd.read_csv(self.central_test_raw_file, sep=r'\s+', engine='c')
        X_central = central_df.iloc[:, 6:].to_numpy(dtype=np.float32)
        
        # 2. Extract Population Statistics
        global_mean = np.nanmean(X_central, axis=0)
        global_std = np.nanstd(X_central, axis=0)
        global_std[global_std == 0] = 1.0  # Prevent divide-by-zero on invariant SNPs

        # 3. Standardize and Fit Scikit-Learn PCA
        X_central_imp = np.where(np.isnan(X_central), global_mean, X_central)
        Z_central = (X_central_imp - global_mean) / global_std
        
        logging.info(f"[Server] Fitting Scikit-Learn PCA on Central Reference Panel (N={Z_central.shape[0]})...")
        pca = PCA(n_components=10, random_state=42)
        scores_central = pca.fit_transform(Z_central).astype(np.float32)

        # 4. Save Central Test TSV
        test_res = pd.DataFrame(scores_central, columns=[f"PC{i}_AVG" for i in range(1, 11)])
        test_res.insert(0, "IID", central_df['IID'].astype(str).tolist())
        test_merged = test_res.merge(self.test_metadata, on="IID")
        test_out = self.out_dir / "central" / "test.tsv"
        test_merged.to_csv(test_out, sep="\t", index=False)
        logging.info(f" -> [central] Derived Axes & successfully written {test_out.name} ({len(test_merged)} samples)")

        logging.info("--- Step 7: Projecting Client Biobanks into Global Space ---")
        
        # 5. Project Local Site Datasets (Train & Validation) using the server's axes
        for site, paths in site_raw_paths.items():
            for split_name in ['train', 'val']:
                local_df = pd.read_csv(paths[split_name], sep=r'\s+', engine='c')
                X_local = local_df.iloc[:, 6:].to_numpy(dtype=np.float32)
                
                # Apply Global Parameters to maintain dimensional fidelity
                X_local_imp = np.where(np.isnan(X_local), global_mean, X_local)
                Z_local = (X_local_imp - global_mean) / global_std
                
                # Project via Sklearn
                scores_local = pca.transform(Z_local).astype(np.float32)
                
                res_df = pd.DataFrame(scores_local, columns=[f"PC{i}_AVG" for i in range(1, 11)])
                res_df.insert(0, "IID", local_df['IID'].astype(str).tolist())
                
                merged_df = res_df.merge(self.train_val_metadata, on="IID")
                out_path = self.out_dir / site / f"{split_name}.tsv"
                merged_df.to_csv(out_path, sep="\t", index=False)
                
            logging.info(f" -> [{site}] Successfully projected local genomes into Global PCA space.")

        # 6. Generate Interactive Validation PCA Plot
        try:
            fig = px.scatter(
                test_merged, 
                x="PC1_AVG", 
                y="PC2_AVG", 
                color="super_pop", 
                title="Global Central Test Set — Scikit-Learn Reference Panel Projection",
                labels={"super_pop": "Super Population"}
            )
            plot_path = self.out_dir / "plots" / "federated_pca_test_pc1_pc2.html"
            fig.write_html(str(plot_path))
            logging.info(f"[Plotting] Interactive FedPCA Scatter Plot saved to {plot_path}")
        except Exception as e:
            logging.warning(f"Could not generate plot: {e}")

        logging.info("=== Pipeline Execution Complete! Ready for Federated PyTorch Training ===")

# =====================================================================
# CLI ENTRYPOINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="End-to-End Scikit-Learn Reference Panel PCA Generation.")
    parser.add_argument("--num-sites", type=int, default=4, help="Number of client biobank sites.")
    parser.add_argument("--dominant-share", type=float, default=0.85, help="Degree of demographic skew per site.")
    parser.add_argument("--cache-dir", type=str, default=str(Path.home() / ".cache" / "federated_ancestry"), help="Path to cache raw genotypes.")
    parser.add_argument("--out-dir", type=str, default="../data", help="Output directory for train/val/test splits.")
    args = parser.parse_args()

    orchestrator = FederatedDataOrchestrator(
        cache_dir=Path(args.cache_dir),
        out_dir=Path(args.out_dir),
        num_sites=args.num_sites,
        dominant_share=args.dominant_share
    )
    orchestrator.run_reference_panel_pipeline()

if __name__ == "__main__":
    main()