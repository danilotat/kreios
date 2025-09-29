import time
from multiprocessing import Pool
from functools import partial
from itertools import islice
import torch

from src.dataset import VariantCollector
from src.channels import (BaseQualityEncoder, SequenceEncoder, IntronicEncoder, 
                          ReferenceMatchingEncoder, ReadStrandEncoder, ReadsSupportingAlleleEncoder)
from src.fetch import ReadFetcher
from src.region import VCFRegion, Region
from src.plotter import TensorVisualizer
import sys
import csv
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ])


def process_variant(variant_info, rnaseq_bam, genome, max_reads, destpath: str):
    """
    Worker function to process a single variant. It's spawned in multiple processes

    Args:
        variant_info (tuple): A tuple containing (variant_id, variant_dict).
        rnaseq_bam (str): Path to the RNA-seq BAM file.
        genome (str): Path to the reference genome FASTA file.
        max_reads (int): Maximum number of reads to process.

    Returns:
        tuple: A tuple containing the region string and the generated tensor.
    """
    variant_id, variant_dict = variant_info    
    # each worker has its own fetcher
    all_encoders = [
        BaseQualityEncoder(),
        SequenceEncoder(),
        IntronicEncoder(),
        ReferenceMatchingEncoder(),
        ReadsSupportingAlleleEncoder(),
        ReadStrandEncoder()
    ]
    fetcher = ReadFetcher(rnaseq_bam, max_reads=max_reads, channel_encoders=all_encoders, genome=genome, device='cpu')

    # 2. Extract region information
    region = Region(variant_dict.get('region'))
    ref, alt = variant_id.split(':')[-1].split('>')
    vcf_region = VCFRegion(variant_dict.get('region'), ref, alt)

    # 3. Fetch and process tensors
    region_tensors = fetcher.fetch_tensors(
        region.chromosome, 
        region.start, 
        region.end, 
        vcf_region
    )
    
    fetcher.close() 
    label = variant_dict.get('label')
    single_tensor = torch.nan_to_num(torch.stack(list(region_tensors.values()), dim=0))
    # store the tensor
    tensor_filename = f"{variant_id.replace(':', '_').replace('>', '_')}.pt"
    torch.save(single_tensor, os.path.join(destpath, tensor_filename) )
    return [tensor_filename, label]


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    VCF_FILE = "/CTGlab/projects/ribo/goyal_2023/RNAseq/VCF/nSyn/SRR20649710.nSyn.vcf.gz"
    destpath="/CTGlab/projects/kreios/data/training"
    sname = "SRR20649710"
    os.makedirs(os.path.join(
        destpath, sname
    ), exist_ok=True)
    RIBO_BAM = "/CTGlab/projects/ribo/goyal_2023/ribo/SRR20648996.genome.sorted.bam"
    RNASEQ_BAM = "/CTGlab/projects/ribo/goyal_2023/RNAseq/SRR20649710_GSM6395082.markdup.sorted.bam"
    GENOME = '/CTGlab/db/GRCh38_GIABv3_no_alt_analysis_set_maskedGRC_decoys_MAP2K3_KMT2C_KCNJ18.fasta' 
    MAX_READS = 256
    NUM_PROCESSES = 10 # Set the number of CPUs to use
    logging.info("Collecting variants...")
    collected_vars = VariantCollector(VCF_FILE, RIBO_BAM, GENOME)
    
    # Create a list of tasks for the workers.
    # We use itertools.islice for a clean way to get the first N items.
    tasks = collected_vars.variants.items()
    
    if not tasks:
        logging.warning("No variants found to process.")
        exit()

    logging.info(f"Prepared {len(tasks)} variants for processing across {NUM_PROCESSES} CPUs.")

    # --- Parallel Processing ---
    # Use functools.partial to create a version of our worker function with the
    # constant arguments (file paths, max_reads) already filled in.
    worker_with_args = partial(process_variant, 
                               rnaseq_bam=RNASEQ_BAM, 
                               genome=GENOME, 
                               max_reads=MAX_READS,
                               destpath=os.path.join(
                                destpath, sname))

    start_time = time.time()
    
    # Create a pool of worker processes. The 'with' statement ensures the pool is closed properly.
    with Pool(processes=NUM_PROCESSES) as pool:
        # pool.map distributes the 'tasks' list to the worker function and collects the results.
        results = pool.map(worker_with_args, tasks)
    metadata_filepath = os.path.join(destpath, sname, "metadata.csv")
    with open(metadata_filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file_path', 'label'])
        writer.writerows(results)
    end_time = time.time()
    logging.info("\n--- Parallel processing complete. ---")
    logging.info(f"Total time to generate {len(tasks)} tensors: {end_time - start_time:.2f} seconds")

    # The 'results' variable is now a list of tuples: [(region_str, tensor), (region_str, tensor), ...]
    # You can now use these results for downstream tasks.
    logging.info(f"\nSuccessfully collected {len(results)} tensors.")