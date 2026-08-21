import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
from urllib.request import urlretrieve
from itertools import product

import pandas
import numpy as np
from tqdm import tqdm, trange
from sklearn.model_selection import KFold, train_test_split
import plotly.express as px

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# CONSTANTS
# ==========================================
TG_SUPERPOP_DICT = {
    'ACB': 'AFR', 'ASW': 'AFR', 'ESN': 'AFR', 'GWD': 'AFR', 'LWK': 'AFR', 'MSL': 'AFR', 'YRI': 'AFR', 
    'CLM': 'AMR', 'MXL': 'AMR', 'PEL': 'AMR', 'PUR': 'AMR', 
    'CDX': 'EAS', 'CHB': 'EAS', 'CHS': 'EAS', 'JPT': 'EAS', 'KHV': 'EAS', 
    'CEU': 'EUR', 'FIN': 'EUR', 'GBR': 'EUR', 'IBS': 'EUR', 'TSI': 'EUR', 
    'BEB': 'SAS', 'GIH': 'SAS', 'ITU': 'SAS', 'PJL': 'SAS', 'STU': 'SAS'
}

PCA_EXTENSIONS = {
    'allele': '.eigenvec.allele',
    'eigenvec': '.eigenvec',
    'counts': '.acount',
    'eigenval': '.eigenval',
    'sscore': '.sscore'
}

# ==========================================
# DATACLASSES
# ==========================================
@dataclass
class CacheArgs:
    path: Path = Path.home() / '.cache' / 'trustworthy_fed_ai'
    num_folds: int = 2  # Changed to 2 folds

@dataclass
class SourceArgs:
    link: Optional[str] = None

@dataclass
class QCArgs:
    sample: Dict[str, str] = field(default_factory=dict)
    variant: Dict[str, str] = field(default_factory=dict)

@dataclass
class SplitArgs:
    num_folds: int = 2  # Changed to 2 folds

@dataclass
class PCAArgs:
    n_components: int = 10

@dataclass
class GlobalArgs:
    cache: CacheArgs
    source: SourceArgs
    qc: QCArgs
    split: SplitArgs
    pca: PCAArgs

# ==========================================
# UTILS & CACHE
# ==========================================
def run_plink(args_list: List[str], args_dict: dict = None):
    """Runs plink 2.0 with specified args. Requires plink2 in PATH."""
    lst = [[k, str(v)] for k, v in args_dict.items()] if args_dict is not None else []
    plink_args = ['plink2'] + args_list + [x for xs in lst for x in xs]
    logging.info(f"Running PLINK command: {' '.join(plink_args)}")
    
    plink = subprocess.run(plink_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if plink.returncode != 0:
        logging.error(plink.stdout.decode('utf-8'))
        raise RuntimeError(plink.stderr.decode('utf-8'))

class FileCache:
    def __init__(self, args: CacheArgs) -> None:
        self.root = Path(args.path)
        self.root.mkdir(parents=True, exist_ok=True)
        for subdir in ['ids', 'phenotypes', 'genotypes', 'plots', 'checkpoints']:
            (self.root / subdir).mkdir(exist_ok=True)
            for fold in range(args.num_folds):
                (self.root / subdir / f'fold_{fold}').mkdir(exist_ok=True)

        self.num_folds = args.num_folds
        self.genotype_stem = 'genotype'  
            
    def vcf(self) -> Tuple[Path, Path]:
        return self.root / 'affymetrix.vcf.gz', self.root / 'affymetrix.vcf.gz.tbi'
    
    def keep_samples_path(self) -> Path:
        return self.root / 'keep.samples'
            
    def ids_path(self, fold_index: int = None, part: str = None) -> Path:
        if fold_index is None and part is None:
            return self.root / 'genotypes' / 'genotype.psam'
        elif fold_index is None:
            return self.root / 'ids' / f'{part}_ids.tsv'
        return self.root / 'ids' / f'fold_{fold_index}' / f'{part}_ids.tsv'
        
    def phenotype_path(self, fold_index: int = None, part: str = None) -> Path:
        if fold_index is None and part is None:
            return self.root / 'phenotypes' / 'phenotypes.tsv'
        elif fold_index is None:
            return self.root / 'phenotypes' / f'{part}_phenotypes.tsv'
        return self.root / 'phenotypes' / f'fold_{fold_index}' / f'{part}_phenotypes.tsv'
        
    def pfile_path(self, fold_index: int = None, part: str = None) -> Path:
        stem = self.genotype_stem
        if fold_index is None and part is None:
            return self.root / 'genotypes' / stem
        elif fold_index is None:
            return self.root / 'genotypes' / f'{part}_{stem}'
        return self.root / 'genotypes' / f'fold_{fold_index}' / f'{part}_{stem}'
        
    def pca_path(self, fold_index: int = None, part: str = None, _type: str = 'allele') -> Path:
        stem = self.genotype_stem
        if fold_index is None and part is None:
            return self.root / 'genotypes' / f'{stem}{PCA_EXTENSIONS[_type]}' 
        elif fold_index is None:
            return self.root / 'genotypes' / f'{part}_{stem}{PCA_EXTENSIONS[_type]}'
        return self.root / 'genotypes' / f'fold_{fold_index}' / f'{part}_{stem}{PCA_EXTENSIONS[_type]}'
        
    def pca_plot_path(self, fold_index: int = None, part: str = None, pc_x: int = 1, pc_y: int = 2) -> Path:
        if fold_index is None and part is None:
            return self.root / 'plots' / f'pca_pc{pc_x}_pc{pc_y}.html' 
        elif fold_index is None:
            return self.root / 'plots' / f'{part}_pca_pc{pc_x}_pc{pc_y}.html'
        return self.root / 'plots' / f'fold_{fold_index}' / f'{part}_pca_pc{pc_x}_pc{pc_y}.html'
class DownloadProgressBar(tqdm):
    """Tracks urlretrieve downloads with dynamic byte units and correct deltas."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)
# ==========================================
# MODULES (Download, QC, Split, PCA)
# ==========================================
class TGDownloader:
    def __init__(self, args: SourceArgs) -> None:
        self.args = args
        self.affymetrix_link = "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/ALL.wgs.nhgri_coriell_affy_6.20140825.genotypes_has_ped.vcf.gz"
        self.panel_link = "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/supporting/hd_genotype_chip/affy_samples.20141118.panel"
        
    def _download_file(self, link: str, output_path: Path) -> None:
            if not output_path.exists():
                with DownloadProgressBar(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=f'Downloading {output_path.name}') as pbar:
                    urlretrieve(link, output_path, reporthook=pbar.update_to)
                logging.info(f'Downloaded {link} to {output_path}')            
            else:
                logging.info(f'File {output_path} already exists')
    
    def _create_keep_samples_file(self, cache: FileCache):
        sf_path = cache.keep_samples_path()
        phenotypes = pandas.read_table(cache.phenotype_path())
        to_keep = phenotypes.loc[phenotypes['pop'].isin(TG_SUPERPOP_DICT), ['sample']]
        to_keep.to_csv(sf_path, sep='\t', index=False)
        
    def _convert_to_pfile(self, cache: FileCache):
        run_plink(
            args_list=['--make-pgen'],
            args_dict={
                '--vcf': str(cache.vcf()[0]),
                '--keep': str(cache.keep_samples_path()),
                '--out': str(cache.pfile_path())
            }
        )
    
    def fit_transform(self, cache: FileCache) -> None:
        vcf, tbi = cache.vcf()
        self._download_file(self.affymetrix_link, vcf)
        self._download_file(self.affymetrix_link + '.tbi', tbi)
        self._download_file(self.panel_link, cache.phenotype_path())
        
        self._create_keep_samples_file(cache)
        self._convert_to_pfile(cache)

class QC:
    def __init__(self, qc_config: Dict) -> None:
        self.qc_config = qc_config
    
    def fit_transform(self, cache: FileCache) -> None:
        qc_path = str(cache.pfile_path()) + "_qc"
        run_plink(
            args_list=['--pfile', str(cache.pfile_path()), '--make-pgen'],
            args_dict={
                '--out': qc_path,
                '--set-missing-var-ids': '@:#:$r:$a',
                **self.qc_config
            }
        )
        # Update cache stem cleanly just once
        cache.genotype_stem = "genotype_qc"

class FoldSplitter:
    def __init__(self, args: SplitArgs) -> None:
        self.args = args
        
    def _split_ids(self, cache: FileCache, y: pandas.Series = None, random_state: int = 34) -> None:
        ids = pandas.read_table(cache.ids_path()).rename(columns={'#IID': 'IID'}).filter(['FID', 'IID'])
        indices = np.arange(ids.shape[0])
        
        if self.args.num_folds <= 1:
            train_indices, val_test_indices = train_test_split(indices, train_size=0.8, random_state=random_state)
            val_indices, test_indices = train_test_split(val_test_indices, train_size=0.5, random_state=random_state)

            for part_indices, part in zip([train_indices, val_indices, test_indices], ['train', 'val', 'test']):
                out_path = cache.ids_path(0, part)
                ids.iloc[part_indices, :].to_csv(out_path, sep='\t', index=False)
            return None
        
        kfsplit = KFold(n_splits=self.args.num_folds, shuffle=True, random_state=random_state).split(ids)
            
        for fold_index, (train_val_indices, test_indices) in enumerate(kfsplit):
            train_indices, val_indices = train_test_split(
                train_val_indices, 
                train_size=(self.args.num_folds - 2) / (self.args.num_folds - 1) if self.args.num_folds > 2 else 0.8,
                random_state=random_state
            )

            for part_indices, part in zip([train_indices, val_indices, test_indices], ['train', 'val', 'test']):
                out_path = cache.ids_path(fold_index, part)
                ids.iloc[part_indices, :].to_csv(out_path, sep='\t', index=False)
                
    def _split_genotypes(self, cache: FileCache) -> None:
        base_path = str(cache.pfile_path())
        for fold_index, part in product(range(cache.num_folds), ['train', 'val', 'test']):
            run_plink(
                args_list=['--make-pgen'],
                args_dict={
                    '--pfile': base_path,
                    '--keep': str(cache.ids_path(fold_index, part)),
                    '--out': str(cache.pfile_path(fold_index, part))
                }
            )
    
    def _split_phenotypes(self, cache: FileCache) -> None:
        phenotype = pandas.read_table(cache.phenotype_path(), names=['IID', 'ancestry', 'in_phase3'])
        for fold_index, part in product(range(cache.num_folds), ['train', 'val', 'test']):
            ids = pandas.read_table(cache.ids_path(fold_index, part))
            fold_phenotype = phenotype.merge(ids, how='inner', on='IID')[['IID', 'ancestry']]
            fold_phenotype.to_csv(cache.phenotype_path(fold_index, part), sep='\t', index=False)
    
    def fit_transform(self, cache: FileCache) -> None:
        self._split_ids(cache)
        self._split_genotypes(cache)
        self._split_phenotypes(cache)

class PCA:
    def __init__(self, args: PCAArgs) -> None:
        self.args = args

    def fit(self, cache: FileCache) -> None:
        for fold in trange(cache.num_folds, desc='PCA on fold', unit='fold'):
            run_plink(
                args_list=['--pca', 'allele-wts', str(self.args.n_components)], 
                args_dict={
                    '--pfile': str(cache.pfile_path(fold, 'train')),
                    '--freq': 'counts', 
                    '--out': str(cache.pfile_path(fold, 'train'))
                }
            )
    
    def transform(self, cache: FileCache) -> None:
        for fold in trange(cache.num_folds, desc='PCA projection', unit='fold'):
            for part in ['train', 'val', 'test']:
                run_plink(
                    # Removed --mac 1 to prevent test set variance drops
                    args_list=['--score', str(cache.pca_path(fold, 'train', 'allele')), '2', '5', 'header-read', 'no-mean-imputation', 'variance-standardize'],
                    args_dict={
                        '--pfile': str(cache.pfile_path(fold, part)),
                        '--read-freq': str(cache.pca_path(fold, 'train', 'counts')), # THIS FIXES THE SILENT CRASH
                        '--score-col-nums': f'6-{6+self.args.n_components - 1}',
                        '--out': str(cache.pfile_path(fold, part))
                    }
                )
                self.pc_scatterplot(cache, fold, part)

    def pc_scatterplot(self, cache: FileCache, fold: int, part: str) -> None:
        """ Visualises eigenvector with scatterplot """
        try:
            eigenvec = pandas.read_table(cache.pca_path(fold, part, 'sscore'))
            # Dynamically grab the exact column names PLINK generated
            pc_cols = [c for c in eigenvec.columns if 'PC1' in c or 'PC2' in c]
            if len(pc_cols) < 2:
                logging.warning(f"Could not find PC1/PC2 columns. Skipping plot.")
                return
            
            pc1_col, pc2_col = pc_cols[0], pc_cols[1]
            eigenvec = eigenvec[['#IID', pc1_col, pc2_col]]
            
            tg_df = pandas.read_table(cache.phenotype_path(fold, part))
            eigenvec = pandas.merge(eigenvec, tg_df, left_on='#IID', right_on='IID')
            eigenvec['ethnic_background_name'] = eigenvec['ancestry'].replace(TG_SUPERPOP_DICT)
            
            fig = px.scatter(eigenvec, x=pc1_col, y=pc2_col, color='ethnic_background_name')
            fig.write_html(cache.pca_plot_path(fold, part))
        except Exception as e:
            logging.warning(f"Failed to generate plot for fold {fold} {part}: {e}")


# ==========================================
# ORCHESTRATOR
# ==========================================
class StandaloneDataPreparer:
    """Replaces GlobalAncestry just for the prepare() phase."""
    def __init__(self, args: GlobalArgs) -> None:
        if args.cache.path is None or str(args.cache.path) == '':
            args.cache.path = Path.home() / '.cache' / 'trustworthy_fed_ai'
        
        self.args = args
        self.cache = FileCache(args.cache)
        self.tg_downloader = TGDownloader(args.source)
        self.sample_splitter = FoldSplitter(args.split)
        self.pca = PCA(args.pca)
        
    def prepare(self) -> None:
        logging.info('Preparing data for global ancestry inference (Standalone)')
        
        self.tg_downloader.fit_transform(self.cache)
        
        # COMBINED QC (Variant + Sample) into one pass to fix _qc_qc bug
        combined_qc_config = {**self.args.qc.variant, **self.args.qc.sample}
        logging.info(f'Running combined QC with {combined_qc_config}')
        combined_qc = QC(combined_qc_config)
        combined_qc.fit_transform(self.cache)
        
        logging.info('Splitting into train, val and test datasets')
        self.sample_splitter.fit_transform(self.cache)
        
        logging.info(f'Running PCA with {self.pca.args.n_components} components')
        self.pca.fit(self.cache)
        self.pca.transform(self.cache)
        
        logging.info('Data preparation finished.')

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # Define Standard Configuration (Reduced to 2 Folds)
    args = GlobalArgs(
        cache=CacheArgs(path=Path.home() / '.cache' / 'trustworthy_fed_ai', num_folds=2),
        source=SourceArgs(),
        qc=QCArgs(
            variant={'--geno': '0.1', '--maf': '0.05', '--hwe': '1e-6'}, 
            sample={'--mind': '0.1'}
        ),
        split=SplitArgs(num_folds=2),
        pca=PCAArgs(n_components=10)
    )

    preparer = StandaloneDataPreparer(args)
    preparer.prepare()