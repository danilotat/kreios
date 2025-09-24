import pysam
import numpy as np
import torch
import os
import logging
import sys
from genome import retrieve_sequence 
from region import Region, VCFRegion
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ])

@dataclass
class VariantCallData:
    """
    A simple container for the variant information needed by certain encoders.
    This mimics the information available in a DeepVariantCall object.
    """
    # Maps an alternate allele string to its frequency (float).
    # from dv_call.allele_frequency()
    allele_frequencies: Dict[str, float] = field(default_factory=dict)

    # Maps an alternate allele string to a list of read names supporting it.
    # from dv_call.allele_support()
    allele_support: Dict[str, List[str]] = field(default_factory=dict)

    # The specific alternate alleles to be considered for generating the image.
    alt_alleles: List[str] = field(default_factory=list)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stderr)
    ])

class MaskedRead:
    """
    Processes a single pysam.AlignedSegment to create padded numpy arrays
    for its sequence and quality scores, accounting for introns.
    """
    def __init__(self, read: pysam.AlignedSegment, seq_padding_token='-', quality_padding_value=np.nan):
        self.read = read
        self.start = self.read.reference_start
        self.end = self.read.reference_end
        self._reference_positions = np.arange(self.start, self.end)
        self.padded_arrays = {}
        aligned_positions = self.read.get_reference_positions(full_length=False)
        query_sequence = self.read.query_sequence
        query_base_qualities = self.read.query_qualities
        query_alignment_qualities = self.read.query_alignment_qualities
        pos_to_query_index = {pos: i for i, pos in enumerate(aligned_positions)}
        padded_sequence = np.full(len(self._reference_positions), fill_value=seq_padding_token, dtype='<U1')
        padded_base_qualities = np.full(len(self._reference_positions), fill_value=quality_padding_value, dtype=np.float32)
        padded_alignment_qualities = np.full(len(self._reference_positions), fill_value=quality_padding_value, dtype=np.float32)
        for i, ref_pos in enumerate(self._reference_positions):
            if ref_pos in pos_to_query_index:
                query_idx = pos_to_query_index[ref_pos]
                if query_idx < len(query_sequence):
                    padded_sequence[i] = query_sequence[query_idx]
                if query_idx < len(query_base_qualities):
                    padded_base_qualities[i] = query_base_qualities[query_idx]
                if query_idx < len(query_alignment_qualities):
                    padded_alignment_qualities[i] = query_alignment_qualities[query_idx]
        self.padded_arrays['sequence'] = padded_sequence
        self.padded_arrays['base_qualities'] = padded_base_qualities
        self.padded_arrays['alignment_qualities'] = padded_alignment_qualities

    def get_padded_array(self, attribute: str):
        if attribute not in self.padded_arrays:
            raise ValueError(f"Attribute '{attribute}' not available.")
        return self.padded_arrays[attribute]

class ReadFetcher:
    """
    Fetches reads from a BAM file and uses a set of ChannelEncoders to
    generate a dictionary of fixed-size, centered, multi-channel torch.Tensors.
    """
    def __init__(self, bam_file: str, max_reads: int, channel_encoders: list, genome: str,  device='cpu'):
        """
        Args:
            bam_file (str): Path to the BAM file.
            max_reads (int): The fixed number of reads for the tensor's height.
            channel_encoders (list[ChannelEncoder]): A list of configured encoder
                objects that will each produce one tensor.
            genome (str): Path to the reference genome FASTA file.
            device (str): The device for tensor creation ('cpu' or 'cuda').
        """
        if max_reads <= 0:
            raise ValueError("max_reads must be a positive integer.")
        self.bam_file = bam_file
        self.bam = pysam.AlignmentFile(bam_file, "rb")
        self.max_reads = max_reads
        self.channel_encoders = channel_encoders
        self.genome = genome
        if not os.path.exists(genome + ".fai"):
            logging.info(f"Index file for genome {genome} not found. Creating index...")
            pysam.faidx(genome)
        self.device = device

    def fetch_tensors(self, chromosome: str, start: int, end: int, variant: VCFRegion) -> dict[str, torch.Tensor]:
        """
        Fetches reads and returns a dictionary of fixed-size torch.Tensors,
        one for each configured ChannelEncoder. The read data is centered
        within the `max_reads` dimension.
        """
        region_obj = Region(f"{chromosome}:{start}-{end}")
        ref_seq = retrieve_sequence(self.genome, region_obj)  
        reads_iterator = self.bam.fetch(
            chromosome, start, end)
        processed_reads = [
            MaskedRead(read, seq_padding_token='-', quality_padding_value=np.nan)
            for read in reads_iterator
            if (not read.is_unmapped and 
                not read.is_secondary and
                not read.is_supplementary and
                not read.is_duplicate)
        ]

        num_reads = len(processed_reads)
        reads_to_process = []
        start_row = 0

        if num_reads > self.max_reads:
            logging.warning(
                f"Found {num_reads} reads, which is more than max_reads={self.max_reads}. "
                f"Taking the central {self.max_reads} reads."
            )
            crop_start = (num_reads - self.max_reads) // 2
            reads_to_process = processed_reads[crop_start : crop_start + self.max_reads]
        else:
            reads_to_process = processed_reads
            start_row = (self.max_reads - num_reads) // 2
        
        region_length = end - start
        tensor_shape = (self.max_reads, region_length)
        tensors = {}

        # Delegate tensor creation to each encoder
        for encoder in self.channel_encoders:
            tensors[encoder.name] = encoder(
                reads_to_process=reads_to_process,
                start_row=start_row,
                tensor_shape=tensor_shape,
                region_start=start,
                device=self.device,
                reference_sequence=ref_seq,
                variant=variant)
            
        return tensors


if __name__ == "__main__":
    from plotter import TensorVisualizer
    pos = 'chr5:140561643-140561644'
    region = 'chr5:140561543-140561743'  # 20bp context on each side
    bam = "/CTGlab/projects/ribo/goyal_2023/RNAseq/SRR20649710_GSM6395082.markdup.sorted.bam"
    genome = ""
    plotter = TensorVisualizer()
    fetcher = ReadFetcher(bam, max_reads=256)
    chromosome, positions = region.split(':')
    start, end = map(int, positions.split('-'))
    reads = fetcher.fetch_tensors(chromosome, start, end, genome=genome)
    print(reads['sequence'].shape, reads['base_qualities'].shape, reads['alignment_qualities'].shape)
    print(reads['sequence'].shape == reads['base_qualities'].shape == reads['alignment_qualities'].shape)
    fig = plotter.plot(reads, chromosome, start, end)	
    fig.savefig("example_plot.png")
