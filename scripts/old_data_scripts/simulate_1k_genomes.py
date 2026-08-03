# simulate_1k_genomes.py
import os
import numpy as np
import pandas as pd

os.makedirs("./local_biobanks/site_a", exist_ok=True)
os.makedirs("./local_biobanks/site_b", exist_ok=True)

SUPERPOPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]

def generate_biobank_data(output_dir, seed, num_samples, heavy_pops):
    np.random.seed(seed)
    
    # 1. Generate Sample IDs
    sample_ids = [f"HG{10000 + i}" for i in range(num_samples)]
    
    # 2. Distribute populations (Simulating realistic Non-IID clinical site bias)
    # Different biobanks naturally have different ethnic/ancestral compositions
    probabilities = [0.6 if pop in heavy_pops else 0.1 for pop in SUPERPOPS]
    probabilities = np.array(probabilities) / sum(probabilities)
    assigned_pops = np.random.choice(SUPERPOPS, size=num_samples, p=probabilities)
    
    # 3. Simulate Top 20 Principal Components
    # Human population groups form tight, mathematically separable clusters on the top PCA axes
    features = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, 20))
    for i, pop in enumerate(assigned_pops):
        pop_index = SUPERPOPS.index(pop)
        # Shift the top 4 components based on ancestry to create distinct clusters
        features[i, :4] += pop_index * 2.5 
    
    # 4. Save the PLINK2 format .sscore file
    sscore_cols = ["#IID"] + [f"PC{i+1}_Avg" for i in range(20)]
    sscore_df = pd.DataFrame(columns=sscore_cols)
    sscore_df["#IID"] = sample_ids
    for i in range(20):
        sscore_df[f"PC{i+1}_Avg"] = features[:, i]
    
    sscore_df.to_csv(os.path.join(output_dir, "genotypes.sscore"), sep="\t", index=False)
    
    # 5. Save the Ancestry ground truth file
    pheno_df = pd.DataFrame({"#IID": sample_ids, "superpop": assigned_pops})
    pheno_df.to_csv(os.path.join(output_dir, "ancestry.tsv"), sep="\t", index=False)
    
    print(f"Biobank created at {output_dir} containing {num_samples} records.")

# Site A is primarily European and South Asian heavy
generate_biobank_data("./local_biobanks/site_a", seed=42, num_samples=400, heavy_pops=["EUR", "SAS"])
# Site B is primarily African and East Asian heavy
generate_biobank_data("./local_biobanks/site_b", seed=43, num_samples=350, heavy_pops=["AFR", "EAS"])