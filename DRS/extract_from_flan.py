import pandas as pd
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
df = df[df['super_pop'] != 'UNKNOWN']

# 3. Partition rows into isolated Site A and Site B datasets
site_a_data = df[df['super_pop'].isin(['AFR', 'EUR'])]
site_b_data = df[df['super_pop'].isin(['EAS', 'SAS'])]

# Define localized storage destinations inside your scratch repository workspace
os.makedirs("local_biobanks/site_a", exist_ok=True)
os.makedirs("local_biobanks/site_b", exist_ok=True)

# 4. Save formatted target files
site_a_data.to_csv("local_biobanks/site_a/genotypes.sscore", sep="\t", index=False)
site_a_data[['IID', 'super_pop']].to_csv("local_biobanks/site_a/ancestry.tsv", sep="\t", index=False)

site_b_data.to_csv("local_biobanks/site_b/genotypes.sscore", sep="\t", index=False)
site_b_data[['IID', 'super_pop']].to_csv("local_biobanks/site_b/ancestry.tsv", sep="\t", index=False)

print("\n--- Split Generation Successful ---")
print(f"Staged Real Site A Dataset (AFR/EUR): {site_a_data.shape[0]} individuals.")
print(f"Staged Real Site B Dataset (EAS/SAS): {site_b_data.shape[0]} individuals.")