"""
This module provides utility functions for tasks such as interacting with
external tools and performing statistical tests.
"""

import logging
import os
import subprocess
from typing import List, Tuple
import numpy as np
from scipy.stats import ranksums


def build_jellyfish_index(
    fastq: str, kmer_size: int, outfile: str, threads: int = 8
) -> str:
    """
    Builds a jellyfish index from a gzipped fastq file.
    """
    if os.path.exists(outfile):
        logging.info(f"Jellyfish index {outfile} already exists. Skipping.")
        return outfile
    
    logging.info(f"Building Jellyfish index from: {fastq}")
    
    # Simple one-liner: zcat pipes directly to jellyfish
    cmd = f"zcat {fastq} | jellyfish count -m {kmer_size} -s 100M -t {threads} -o {outfile}"
    
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Jellyfish failed: {e.stderr}")
        raise RuntimeError("Jellyfish index creation failed.")
    
    if not os.path.exists(outfile):
        raise FileNotFoundError(f"Jellyfish index not created: {outfile}")
    
    logging.info(f"Successfully created Jellyfish index: {outfile}")
    return outfile


def calculate_rank_sum_test(
    alternate_dist: List[int], reference_dist: List[int]
) -> Tuple[float, float]:
    """Calculates the rank sum test for two distributions."""
    if not alternate_dist or not reference_dist:
        return np.nan, np.nan

    stat, pvalue = ranksums(x=alternate_dist, y=reference_dist)
    return round(stat, 3), round(pvalue, 5)
