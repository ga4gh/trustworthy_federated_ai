# merge_flan_data_4sites.py
"""
Like merge_flan_data.py, but:
  1. Pools ALL folds (fold_0..fold_4) of the FLAN sscore cache instead of just
     fold_0, deduplicating on IID in case the folds overlap (k-fold CV splits
     are not guaranteed disjoint -- safer to dedupe than assume).
  2. Writes ONE unified_data.tsv per site (genotypes + ancestry already merged),
     same as merge_flan_data.py's output shape.
  3. Partitions into 4 sites, each DOMINANT in exactly one superpopulation
     (not two, like the earlier extract_from_flan_4sites.py) -- this matches
     real biobank/cohort skew far better: a site is overwhelmingly local-
     population, with everything else present only as a thin trickle.
     AMR has no natural "home" continent in this 4-site layout, so it's
     spread evenly across all 4 sites as a minority class everywhere.

Usage:
    python merge_flan_data_4sites.py
"""
import os
import glob
import pandas as pd
import numpy as np

FOLD_GLOB = "/home/viditkh/.cache/deep_ancestry/genotypes/fold_*/train_genotype.sscore"
METADATA_URL = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20130606_sample_info/20130606_sample_info.txt"

POPULATION_TO_SUPERPOP = {
    'CEU': 'EUR', 'TSI': 'EUR', 'FIN': 'EUR', 'GBR': 'EUR', 'IBS': 'EUR',
    'YRI': 'AFR', 'LWK': 'AFR', 'GWD': 'AFR', 'MSL': 'AFR', 'ESN': 'AFR', 'ASW': 'AFR', 'ACB': 'AFR',
    'CHB': 'EAS', 'JPT': 'EAS', 'CHS': 'EAS', 'CDX': 'EAS', 'KHV': 'EAS',
    'GIH': 'SAS', 'PJL': 'SAS', 'BEB': 'SAS', 'STU': 'SAS', 'ITU': 'SAS',
    'MXL': 'AMR', 'PUR': 'AMR', 'CLM': 'AMR', 'PEL': 'AMR',
}

SUPERPOPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]

# Each of these 4 sites is dominant in exactly one superpop. AMR is intentionally
# left out of this map -- it has no single home continent here, so it gets
# spread evenly across all 4 sites further down instead.
SITE_DOMINANT_POP = {
    "site_a": "EUR",
    "site_b": "AFR",
    "site_c": "EAS",
    "site_d": "SAS",
}
# Fraction of a dominant pop's individuals that land in their home site. The
# remainder is split evenly across the other 3 sites as minority presence --
# kept nonzero everywhere on purpose, real-world cohorts always have a little
# of everything via immigration, multi-ethnic recruitment, etc.
DOMINANT_SHARE = 0.85
SEED = 7


def load_all_folds():
    fold_paths = sorted(glob.glob(FOLD_GLOB))
    if not fold_paths:
        raise FileNotFoundError(f"No fold files matched: {FOLD_GLOB}")
    print(f"Found {len(fold_paths)} fold file(s):")
    for p in fold_paths:
        print(f"  - {p}")

    frames = []
    for p in fold_paths:
        fold_df = pd.read_csv(p, sep=r'\s+')
        fold_df.rename(columns={'#IID': 'IID', '#FID': 'FID'}, inplace=True)
        frames.append(fold_df)

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset='IID', keep='first').reset_index(drop=True)
    after = len(combined)
    if before != after:
        print(f"Deduplicated {before - after} repeated IIDs across folds "
              f"(folds overlapped) -> {after} unique individuals")
    else:
        print(f"No duplicate IIDs across folds -> {after} unique individuals")
    return combined


def main():
    df = load_all_folds()

    print("Downloading population reference panel metadata maps...")
    meta_df = pd.read_csv(METADATA_URL, sep='\t')
    pop_map = dict(zip(meta_df['Sample'], meta_df['Population']))

    df['super_pop'] = [
        POPULATION_TO_SUPERPOP.get(pop_map.get(iid), 'UNKNOWN') for iid in df['IID']
    ]
    df = df[df['super_pop'] != 'UNKNOWN'].reset_index(drop=True)
    print(f"\nPooled (all folds) labeled dataset: {len(df)} individuals")
    print("Class counts:", df['super_pop'].value_counts().reindex(SUPERPOPS, fill_value=0).to_dict())

    # Assign each individual to exactly one site. Pops with a home site send
    # DOMINANT_SHARE of their individuals there, the rest split evenly across
    # the other 3 sites. AMR (no home site) splits evenly across all 4.
    rng = np.random.RandomState(SEED)
    site_names = list(SITE_DOMINANT_POP.keys())
    pop_to_home_site = {pop: site for site, pop in SITE_DOMINANT_POP.items()}

    assigned_site = []
    for pop in df['super_pop']:
        home_site = pop_to_home_site.get(pop)
        if home_site is not None:
            if rng.rand() < DOMINANT_SHARE:
                assigned_site.append(home_site)
            else:
                others = [s for s in site_names if s != home_site]
                assigned_site.append(others[rng.randint(len(others))])
        else:
            assigned_site.append(site_names[rng.randint(len(site_names))])
    df['assigned_site'] = assigned_site

    print("\n--- Split Generation Successful ---")
    for site in site_names:
        site_data = df[df['assigned_site'] == site].drop(columns=['assigned_site'])
        out_dir = f"local_biobanks/{site}"
        os.makedirs(out_dir, exist_ok=True)

        out_path = f"{out_dir}/unified_data.tsv"
        site_data.to_csv(out_path, sep="\t", index=False)

        counts = site_data['super_pop'].value_counts().reindex(SUPERPOPS, fill_value=0)
        print(f"Staged Real {site} (dominant: {SITE_DOMINANT_POP[site]}): "
              f"{site_data.shape[0]} individuals -> {out_path} | " +
              " ".join(f"{p}={counts[p]}" for p in SUPERPOPS))


if __name__ == "__main__":
    main()