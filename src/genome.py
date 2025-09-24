import pysam
from typing import List
from region import Region
import torch


def retrieve_sequence(genome: str, interval: Region) -> str:
    """
    This function retrieves the sequence for the given interval from the genome.
    """
    genome = pysam.FastaFile(genome)
    print(interval)
    sequence = genome.fetch(
        interval.chromosome,
        interval.start,
        interval.end)
    return sequence

    
