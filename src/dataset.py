import pysam
import cyvcf2
from collections import namedtuple
import logging
import sys
import torch
import os
import pandas as pd
from torch.utils.data import Dataset

class VariantTensorDataset(Dataset):
    """
    Pytorch dataset for loading tensors and the according metadata file. 
    Attributes:
        dataset_dir (str): The root directory containing the dataset.
        metadata (pd.DataFrame): A DataFrame containing metadata about the dataset, including
            file paths and labels.
    Methods:
        __len__():
            Returns the total number of samples in the dataset.
        __getitem__(idx):
            Retrieves the tensor and label for the given index.
    """
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        self.metadata = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get the file path and label for the requested index
        tensor_filename = self.metadata.iloc[idx]['file_path']
        label = self.metadata.iloc[idx]['label']

        # Construct the full path to the tensor file
        tensor_path = os.path.join(self.dataset_dir, tensor_filename)
        
        # Load the single tensor from disk
        tensor = torch.load(tensor_path)
        
        # Convert label to a tensor if needed
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return tensor, label_tensor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ])

coverage = namedtuple('Coverage', ['A', 'C', 'G', 'T'])

class VariantExtended():
    def __init__(self, variant: cyvcf2.Variant, genome: str,  bam_file: str, flanking=128):
        self.variant = variant
        self.flanking = flanking
        self.full_region = f"{variant.CHROM}:{variant.POS - (self.flanking / 2)}-{self.variant.POS + (self.flanking / 2)}" 
        self.identifier = f"{variant.CHROM}:{variant.POS}:{variant.REF}>{variant.ALT[0]}"
        self.bam_file = pysam.AlignmentFile(bam_file, 'rb')
        self.genome = pysam.FastaFile(genome)
        self.reference_sequence = self._get_sequence()
        self.has_support = self._get_support()
    
    def _get_sequence(self):
        sequence = self.genome.fetch(
            self.variant.CHROM, self.variant.POS - (self.flanking / 2), self.variant.POS + (self.flanking / 2)
        )
        return sequence

    def _get_support(self):
        # fetch reads in position and check if it has support in the matched aligned bam. 
        bam_coverage = self.bam_file.count_coverage(
            self.variant.CHROM, self.variant.POS, self.variant.POS + 1, read_callback='all'
        )
        # arrays are in order A,C,G,T
        coverage_nt = coverage(*[bam_coverage[i][0] for i in range(4)])
        if getattr(coverage_nt, self.variant.ALT[0]) > 0:
            return True
        else:
            return False 


class VariantCollector():
    """
    @TODO: implement a smart way to iterate here using __next__ and __iter__
    """
    def __init__(self, vcf: str, bam: str, genome: str):
        self.vcf = cyvcf2.VCF(vcf)
        self.bam = bam
        self.genome = genome
        self.variants = self._collect_variants()
    
    def _collect_variants(self):
        collector = {}
        variants = 0
        supported = 0
        not_supported = 0
        for variant in self.vcf:
            variants += 1
            if variant.is_snp: # support just for SNP now
                variantExt = VariantExtended(
                    variant, self.genome, self.bam
                )
                if variantExt.has_support:
                    supported += 1
                else:
                    not_supported +=1
                collector[variantExt.identifier] = {
                    'region': variantExt.full_region,
                    'reference_sequence': variantExt.reference_sequence,
                    'label': 1 if variantExt.has_support else 0  
                }
        logging.info(f"Processed {variants} variants")
        logging.info(f"Found support for {supported}/{variants}")
        logging.info(f"The remaining unsupported are {not_supported}/{variants}")
        return collector

if __name__ == '__main__':
    import sys
    vcf = "/CTGlab/projects/ribo/goyal_2023/RNAseq/VCF/nSyn/SRR20649710.nSyn.vcf.gz"
    bam = "/CTGlab/projects/ribo/goyal_2023/ribo/SRR20648996.genome.sorted.bam"
    genome = '/CTGlab/db/GRCh38_GIABv3_no_alt_analysis_set_maskedGRC_decoys_MAP2K3_KMT2C_KCNJ18.fasta' 
    collected_vars = VariantCollector(vcf, bam, genome)


