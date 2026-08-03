import pandas as pd
import os

# Paths to the raw fold_0 matrix files
sscore_file = "/home/viditkh/.cache/deep_ancestry/genotypes/fold_0/train_genotype.sscore"

print("Parsing features out of fold_0 cache matrix...")
features_df = pd.read_csv(sscore_file, sep=r'\s+')
features_df.rename(columns={'#IID': 'IID', '#FID': 'FID'}, inplace=True, errors='ignore')

# Download the exact cross-referenced population labels 
print("Cross-referencing ancestry mappings...")
metadata_url = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20130606_sample_info/20130606_sample_info.txt"
meta_df = pd.read_csv(metadata_url, sep='\t')

population_to_superpop = {
    'CEU': 'EUR', 'TSI': 'EUR', 'FIN': 'EUR', 'GBR': 'EUR', 'IBS': 'EUR',
    'YRI': 'AFR', 'LWK': 'AFR', 'GWD': 'AFR', 'MSL': 'AFR', 'ESN': 'AFR', 'ASW': 'AFR', 'ACB': 'AFR',
    'CHB': 'EAS', 'JPT': 'EAS', 'CHS': 'EAS', 'CDX': 'EAS', 'KHV': 'EAS',
    'GIH': 'SAS', 'PJL': 'SAS', 'BEB': 'SAS', 'STU': 'SAS', 'ITU': 'SAS',
    'MXL': 'AMR', 'PUR': 'AMR', 'CLM': 'AMR', 'PEL': 'AMR'
}
pop_map = dict(zip(meta_df['Sample'], meta_df['Population']))

super_pops = [population_to_superpop.get(pop_map.get(iid), 'UNKNOWN') for iid in features_df['IID']]
features_df['super_pop'] = super_pops
features_df = features_df[features_df['super_pop'] != 'UNKNOWN']

# Split into Site A and Site B datasets
site_a_df = features_df[features_df['super_pop'].isin(['AFR', 'EUR'])]
site_b_df = features_df[features_df['super_pop'].isin(['EAS', 'SAS'])]

os.makedirs("local_biobanks/site_a", exist_ok=True)
os.makedirs("local_biobanks/site_b", exist_ok=True)

# Write out unified data frames containing features + targets side-by-side
site_a_df.to_csv("local_biobanks/site_a/unified_data.tsv", sep="\t", index=False)
site_b_df.to_csv("local_biobanks/site_b/unified_data.tsv", sep="\t", index=False)

print(f"Staged unified file for Site A: {site_a_df.shape[0]} rows.")
print(f"Staged unified file for Site B: {site_b_df.shape[0]} rows.")