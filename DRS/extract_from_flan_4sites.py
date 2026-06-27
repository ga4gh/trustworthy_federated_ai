# extract_from_flan_4sites.py
"""
Same real-data extraction pipeline as extract_from_flan.py, but partitions
individuals into 4 sites instead of 2, each "heavy" in a different pair of
superpopulations (Non-IID, mirrors the skew style used in simulate_1k_genomes.py)
rather than a hard disjoint split. Disjoint splits (like the original AFR/EUR
vs EAS/SAS 2-site version) silently drop AMR and zero out 3 of 5 classes per
site, which makes per-class FL behavior impossible to study. Here every site
gets a slice of every superpop, just skewed in different directions.
"""
import pandas as pd
import numpy as np
import os

# 1. Point to the exact sscore file sitting in FLAN's fold_0 cache
sscore_path = "/home/viditkh/.cache/deep_ancestry/genotypes/fold_0/train_genotype.sscore"

print(f"Reading real preprocessed features from: {sscore_path}")
# PLINK2 .sscore files are tab/space separated.
# Columns are typically: #FID, IID, ALLELE_CT, NAMED_ALLELE_DOSAGE_SUM, PC1_AVG, PC2_AVG...
df = pd.read_csv(sscore_path, sep=r'\s+')

# Clean up column header string if it contains a leading hash
df.rename(columns={'#IID': 'IID', '#FID': 'FID'}, inplace=True)

# 2. Fetch the true superpopulation ancestry references from the 1000 Genomes project
print("Downloading population reference panel metadata maps...")
metadata_url = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20130606_sample_info/20130606_sample_info.txt"
meta_df = pd.read_csv(metadata_url, sep='\t')

# Lookup map transforming specific 3-letter codes to broad continental classes
population_to_superpop = {
    'CEU': 'EUR', 'TSI': 'EUR', 'FIN': 'EUR', 'GBR': 'EUR', 'IBS': 'EUR',
    'YRI': 'AFR', 'LWK': 'AFR', 'GWD': 'AFR', 'MSL': 'AFR', 'ESN': 'AFR', 'ASW': 'AFR', 'ACB': 'AFR',
    'CHB': 'EAS', 'JPT': 'EAS', 'CHS': 'EAS', 'CDX': 'EAS', 'KHV': 'EAS',
    'GIH': 'SAS', 'PJL': 'SAS', 'BEB': 'SAS', 'STU': 'SAS', 'ITU': 'SAS',
    'MXL': 'AMR', 'PUR': 'AMR', 'CLM': 'AMR', 'PEL': 'AMR'
}
pop_map = dict(zip(meta_df['Sample'], meta_df['Population']))

# Line up labels matching individual sample IDs
super_pops = []
for sample_id in df['IID']:
    specific_pop = pop_map.get(sample_id, None)
    super_pops.append(population_to_superpop.get(specific_pop, 'UNKNOWN'))

df['super_pop'] = super_pops
df = df[df['super_pop'] != 'UNKNOWN'].reset_index(drop=True)

# 3. Partition rows into 4 Non-IID sites. Each site is "heavy" (60% mass) in two
#    superpops and gets the remainder spread thinly across the other three, so no
#    class is ever fully absent from a site -- unlike a disjoint AFR/EUR vs EAS/SAS
#    split, which leaves AMR with zero representation anywhere.
SUPERPOPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
SITE_HEAVY_POPS = {
    "site_a": ["EUR", "SAS"],
    "site_b": ["AFR", "EAS"],
    "site_c": ["AMR", "EUR"],
    "site_d": ["SAS", "AFR"],
}

rng = np.random.RandomState(7)

# Assign every individual a per-site "weight" based on whether their true superpop
# is one of that site's heavy pops, then sample disjoint membership proportional
# to those weights so each individual lands in exactly one site (no leakage across
# sites) while the heavy/light skew still comes through in the final composition.
site_names = list(SITE_HEAVY_POPS.keys())
weights = np.zeros((len(df), len(site_names)))
for j, site in enumerate(site_names):
    heavy = SITE_HEAVY_POPS[site]
    weights[:, j] = df['super_pop'].apply(lambda p: 3.0 if p in heavy else 1.0).values

probs = weights / weights.sum(axis=1, keepdims=True)
assigned_site_idx = np.array([
    rng.choice(len(site_names), p=probs[i]) for i in range(len(df))
])
df['assigned_site'] = [site_names[i] for i in assigned_site_idx]

# 4. Save formatted target files per site, same layout as the original 2-site script
print("\n--- Split Generation Successful ---")
for site in site_names:
    site_data = df[df['assigned_site'] == site].drop(columns=['assigned_site'])
    out_dir = f"local_biobanks/{site}"
    os.makedirs(out_dir, exist_ok=True)

    site_data.to_csv(f"{out_dir}/genotypes.sscore", sep="\t", index=False)
    site_data[['IID', 'super_pop']].to_csv(f"{out_dir}/ancestry.tsv", sep="\t", index=False)

    counts = site_data['super_pop'].value_counts().reindex(SUPERPOPS, fill_value=0)
    print(f"Staged Real {site} (heavy: {SITE_HEAVY_POPS[site]}): "
          f"{site_data.shape[0]} individuals | " +
          " ".join(f"{p}={counts[p]}" for p in SUPERPOPS))
