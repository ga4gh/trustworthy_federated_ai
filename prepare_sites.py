# prepare_sites.py
import os
import pandas as pd
import numpy as np
import json

def process_and_merge(data_dir, split_name, pop_to_idx, pop_to_superpop):
    """Reads PLINK sscore and phenotype tables, maps granular ancestry to superpopulation, and merges."""
    sscore_path = os.path.join(data_dir, f"{split_name}_genotype.sscore")
    pheno_path = os.path.join(data_dir, f"{split_name}_phenotypes.tsv")
    
    if not os.path.exists(sscore_path) or not os.path.exists(pheno_path):
        print(f"Warning: Missing files for {split_name} in {data_dir}/")
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
    
    # Filter down to the exact columns present in the dataset
    feature_cols = [f"PC{i}_AVG" for i in range(1, 11)]
    available_features = [col for col in feature_cols if col in merged.columns]
    keep_cols = ["IID"] + available_features + ["super_pop"]
    
    merged = merged[keep_cols]
    
    return merged.dropna()


def assign_sites(df, site_dominant_pop, dominant_share, seed=42):
    """Assigns individuals to specific sites mimicking real-world biobank skew."""
    rng = np.random.RandomState(seed)
    site_names = list(site_dominant_pop.keys())
    pop_to_home_site = {pop: site for site, pop in site_dominant_pop.items()}

    assigned_site = []
    for pop in df['super_pop']:
        home_site = pop_to_home_site.get(pop)
        if home_site is not None:
            if rng.rand() < dominant_share:
                assigned_site.append(home_site)
            else:
                others = [s for s in site_names if s != home_site]
                assigned_site.append(others[rng.randint(len(others))])
        else:
            assigned_site.append(site_names[rng.randint(len(site_names))])
            
    df['assigned_site'] = assigned_site
    return df


def main():
    DATA_DIR = "."
    
    POPULATION_TO_SUPERPOP = {
        'CEU': 'EUR', 'TSI': 'EUR', 'FIN': 'EUR', 'GBR': 'EUR', 'IBS': 'EUR',
        'YRI': 'AFR', 'LWK': 'AFR', 'GWD': 'AFR', 'MSL': 'AFR', 'ESN': 'AFR', 'ASW': 'AFR', 'ACB': 'AFR',
        'CHB': 'EAS', 'JPT': 'EAS', 'CHS': 'EAS', 'CDX': 'EAS', 'KHV': 'EAS',
        'GIH': 'SAS', 'PJL': 'SAS', 'BEB': 'SAS', 'STU': 'SAS', 'ITU': 'SAS',
        'MXL': 'AMR', 'PUR': 'AMR', 'CLM': 'AMR', 'PEL': 'AMR',
    }
    
    POP_TO_IDX = {"AFR": 0, "AMR": 1, "EAS": 2, "EUR": 3, "SAS": 4}
    SUPERPOPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
    
    SITE_DOMINANT_POP = {
        "site_a": "EUR",
        "site_b": "AFR",
        "site_c": "EAS",
        "site_d": "SAS",
    }
    DOMINANT_SHARE = 0.85
    
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/label_mapping.json", "w") as f:
        json.dump(POP_TO_IDX, f)

    # 1. Process and save Server/Global Test sets directly
    print(f"[Data Pipeline] Processing Global Test Sheet from {DATA_DIR}/...")
    test_df = process_and_merge(DATA_DIR, "test", POP_TO_IDX, POPULATION_TO_SUPERPOP)
    
    os.makedirs("server", exist_ok=True)
    if test_df is not None:
        test_df.to_csv("server/test.tsv", sep="\t", index=False)
        print(f" -> server/test.tsv staged with {len(test_df)} individuals.")

    # 2. Slice and shard Training and Validation data dynamically across 4 biobanks
    print("\n[Data Pipeline] Sharding Training and Validation sets with population skew...")
    train_df = process_and_merge(DATA_DIR, "train", POP_TO_IDX, POPULATION_TO_SUPERPOP)
    val_df = process_and_merge(DATA_DIR, "val", POP_TO_IDX, POPULATION_TO_SUPERPOP)
    
    if train_df is not None:
        train_df = assign_sites(train_df, SITE_DOMINANT_POP, DOMINANT_SHARE, seed=42)
    if val_df is not None:
        val_df = assign_sites(val_df, SITE_DOMINANT_POP, DOMINANT_SHARE, seed=43)
        
    site_names = list(SITE_DOMINANT_POP.keys())
    
    for site in site_names:
        os.makedirs(site, exist_ok=True)
        
        # Output Train
        if train_df is not None:
            site_train = train_df[train_df['assigned_site'] == site].drop(columns=['assigned_site'])
            out_path_train = os.path.join(site, "train.tsv")
            site_train.to_csv(out_path_train, sep="\t", index=False)
            
            counts = site_train['super_pop'].value_counts().reindex(SUPERPOPS, fill_value=0).to_dict()
            print(f" -> {out_path_train} staged | Dominant: {SITE_DOMINANT_POP[site]} | N={len(site_train)} | Counts: {counts}")
            
        # Output Val
        if val_df is not None:
            site_val = val_df[val_df['assigned_site'] == site].drop(columns=['assigned_site'])
            out_path_val = os.path.join(site, "val.tsv")
            site_val.to_csv(out_path_val, sep="\t", index=False)
            print(f" -> {out_path_val} staged | N={len(site_val)}")
            
    print("\n[Data Pipeline] Finished successfully.")

if __name__ == "__main__":
    main()