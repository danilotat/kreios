import bionumpy as bnp
from typing import List
from .region import Region
import torch


def retrieve_context(genome: bnp.Genome, intervals: List[Region], context: int = 20) -> List[str]:
    """
    This function retrieves the sequences for the given intervals from the genome, returning the sequences with added context on both sides.
    NOTE: this could be used more effectively to play with pytorch.
    """
    intervals = [(interval.chromosome, max(0, interval.start - context), interval.end + context) for interval in intervals]
    sequences = genome.read_sequence()[intervals]
    sequences = bnp.as_encoded_array(sequences, bnp.DNAEncoding)
    return sequences 

def to_multi_channel_tensor(sequences: List[str]) -> torch.Tensor:
    """
    Convert a list of DNA sequences to a multi-channel tensor representation.
    """
    
