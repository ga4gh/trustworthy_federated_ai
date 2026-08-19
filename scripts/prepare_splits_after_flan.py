import os
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path

def process_and_merge(cache_dir, fold, split_name, pop_to_idx, pop_to_superpop):
    """Reads PLINK sscore and phenotype tables from the cache directory and merges them."""
    
    sscore_path = os.path.join(cache_dir, "genotypes", f"fold_{fold}", f"{split_name}_genotype_qc.sscore")
    pheno_path = os.path.join(cache_dir, "phenotypes", f"fold_{fold}", f"{split_name}_phenotypes.tsv")
    
    if not os.path.exists(sscore_path) or not os.path.exists(pheno_path):
        print(f"Warning: Missing files for {split_name} in Fold {fold}. Looked for:\n  - {sscore_path}\n  - {pheno_path}")
        return None

    # Handle space/tab delimiters common in PLINK outputs
    df_sscore = pd.read_csv(sscore_path, sep=r'\s+', engine='python')
    df_pheno = pd.read_csv(pheno_path, sep=r'\s+', engine='python')
    
    # Clean up column names in case FLAN/PLINK output #IID instead of IID
    if '#IID' in df_sscore.columns:
        df_sscore.rename(columns={'#IID': 'IID'}, inplace=True)
    if '#IID' in df_pheno.columns:
        df_pheno.rename(columns={'#IID': 'IID'}, inplace=True)
        
    # Merge on Individual ID (IID)
    merged = pd.merge(df_sscore, df_pheno, on="IID")
    
    # Map granular 'ancestry' column to 'super_pop'
    if 'ancestry' in merged.columns:
        merged['super_pop'] = merged['ancestry'].map(pop_to_superpop)
    else:
        print(f"Error: 'ancestry' column missing in {pheno_path}")
        return None
    
    # Drop rows where population mapping is unknown
    merged = merged[merged['super_pop'].notna()]
    
    # Filter down to the exact columns present in the dataset (PC1 to PC10)
    feature_cols = [f"PC{i}_AVG" for i in range(1, 11)]
    available_features = [col for col in feature_cols if col in merged.columns]
    keep_cols = ["IID"] + available_features + ["super_pop"]
    
    merged = merged[keep_cols]
    
    return merged.dropna()


def assign_sites(df, site_dominant_pop, dominant_share, seed=42):
    """Assigns individuals to specific sites mimicking real-world biobank skew."""
    rng = np.random.RandomState(seed)
    site_names = list(site_dominant_pop.keys())
    pop_to_home_site = {pop: [] for pop in set(site_dominant_pop.values())}
    
    # Map which sites act as 'home' for which populations
    for site, pop in site_dominant_pop.items():
        pop_to_home_site[pop].append(site)

    assigned_site = []
    for pop in df['super_pop']:
        home_sites = pop_to_home_site.get(pop, [])
        if home_sites:
            if rng.rand() < dominant_share:
                # Assign to one of the home sites for this population
                assigned_site.append(rng.choice(home_sites))
            else:
                # Assign to a completely random different site
                others = [s for s in site_names if s not in home_sites]
                assigned_site.append(rng.choice(others) if others else rng.choice(site_names))
        else:
            assigned_site.append(rng.choice(site_names))
            
    df['assigned_site'] = assigned_site
    return df


def main():
    parser = argparse.ArgumentParser(description="Dynamically prepare FL client data from PLINK cache.")
    parser.add_argument("--num-sites", type=int, default=4, help="Number of client sites to generate.")
    parser.add_argument("--fold", type=int, default=0, help="Which fold to use from the cache.")
    parser.add_argument("--cache-dir", type=str, default=str(Path.home() / '.cache' / 'trustworthy_fed_ai'), help="Path to the global prepared data cache.")
    parser.add_argument("--out-dir", type=str, default="../data", help="Where to save the output TSVs for the DRS seeder.")
    args = parser.parse_args()

    POPULATION_TO_SUPERPOP = {
        'CEU': 'EUR', 'TSI': 'EUR', 'FIN': 'EUR', 'GBR': 'EUR', 'IBS': 'EUR',
        'YRI': 'AFR', 'LWK': 'AFR', 'GWD': 'AFR', 'MSL': 'AFR', 'ESN': 'AFR', 'ASW': 'AFR', 'ACB': 'AFR',
        'CHB': 'EAS', 'JPT': 'EAS', 'CHS': 'EAS', 'CDX': 'EAS', 'KHV': 'EAS',
        'GIH': 'SAS', 'PJL': 'SAS', 'BEB': 'SAS', 'STU': 'SAS', 'ITU': 'SAS',
        'MXL': 'AMR', 'PUR': 'AMR', 'CLM': 'AMR', 'PEL': 'AMR',
    }
    
    POP_TO_IDX = {"AFR": 0, "AMR": 1, "EAS": 2, "EUR": 3, "SAS": 4}
    SUPERPOPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
    
    # ---------------------------------------------------------
    # DYNAMIC SITE GENERATION
    # ---------------------------------------------------------
    SITE_DOMINANT_POP = {}
    for i in range(1, args.num_sites + 1):
        # Round-robin assign dominant populations to sites (site_1, site_2, etc.)
        pop = SUPERPOPS[(i - 1) % len(SUPERPOPS)]
        SITE_DOMINANT_POP[f"site_{i}"] = pop
        
    DOMINANT_SHARE = 0.85
    
    # Setup output directories for Kubernetes host mounts
    os.makedirs(os.path.join(args.out_dir, "central"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "checkpoints"), exist_ok=True)
    
    with open(os.path.join(args.out_dir, "checkpoints", "label_mapping.json"), "w") as f:
        json.dump(POP_TO_IDX, f)

    # 1. Process and save Server/Global Test sets directly
    print(f"\n[Data Pipeline] Sourcing Global Test Sheet from cache Fold {args.fold}...")
    test_df = process_and_merge(args.cache_dir, args.fold, "test", POP_TO_IDX, POPULATION_TO_SUPERPOP)
    
    if test_df is not None:
        # CHANGED: Now simply named test.tsv
        out_test = os.path.join(args.out_dir, "central", "test.tsv")
        test_df.to_csv(out_test, sep="\t", index=False)
        print(f" -> {out_test} staged with {len(test_df)} individuals.")

    # 2. Slice and shard Training and Validation data dynamically
    print(f"\n[Data Pipeline] Sharding Training and Validation sets across {args.num_sites} dynamic sites...")
    train_df = process_and_merge(args.cache_dir, args.fold, "train", POP_TO_IDX, POPULATION_TO_SUPERPOP)
    val_df = process_and_merge(args.cache_dir, args.fold, "val", POP_TO_IDX, POPULATION_TO_SUPERPOP)
    
    if train_df is not None:
        train_df = assign_sites(train_df, SITE_DOMINANT_POP, DOMINANT_SHARE, seed=42)
    if val_df is not None:
        val_df = assign_sites(val_df, SITE_DOMINANT_POP, DOMINANT_SHARE, seed=43)
        
    for site, dominant_pop in SITE_DOMINANT_POP.items():
        site_dir = os.path.join(args.out_dir, site)
        os.makedirs(site_dir, exist_ok=True)
        
        # Output Train
        if train_df is not None:
            site_train = train_df[train_df['assigned_site'] == site].drop(columns=['assigned_site'])
            # CHANGED: Now simply named train.tsv
            out_path_train = os.path.join(site_dir, "train.tsv")
            site_train.to_csv(out_path_train, sep="\t", index=False)
            
            counts = site_train['super_pop'].value_counts().reindex(SUPERPOPS, fill_value=0).to_dict()
            print(f" -> {out_path_train} staged | Dominant: {dominant_pop} | N={len(site_train)} | Counts: {counts}")
            
        # Output Val
        if val_df is not None:
            site_val = val_df[val_df['assigned_site'] == site].drop(columns=['assigned_site'])
            # CHANGED: Now simply named val.tsv
            out_path_val = os.path.join(site_dir, "val.tsv")
            site_val.to_csv(out_path_val, sep="\t", index=False)
            
    print("\n[Data Pipeline] Dynamic Data Preparation Finished Successfully.")

if __name__ == "__main__":
    main()