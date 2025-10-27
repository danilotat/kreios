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
    """Builds a jellyfish index from a fastq file."""
    if os.path.exists(outfile):
        logging.info(f"File {outfile} already exists.")
        return outfile

    command = [
        "jellyfish",
        "count",
        "-m",
        str(kmer_size),
        "-s",
        "100M",
        "-t",
        str(threads),
        "-C",
        fastq,
        "-o",
        outfile,
    ]
    subprocess.run(command, check=True)

    if not os.path.exists(outfile):
        raise FileNotFoundError(f"Jellyfish index file was not created: {outfile}")

    return outfile


def calculate_rank_sum_test(
    alternate_dist: List[int], reference_dist: List[int]
) -> Tuple[float, float]:
    """Calculates the rank sum test for two distributions."""
    if not alternate_dist or not reference_dist:
        return np.nan, np.nan

    stat, pvalue = ranksums(x=alternate_dist, y=reference_dist)
    return round(stat, 3), round(pvalue, 5)
