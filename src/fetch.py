import pysam
import numpy as np
import torch
import os
import logging
import sys
from .genome import retrieve_sequence 
from .region import Region, VCFRegion
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
    OPTIMIZED: Processes a single pysam.AlignedSegment to create padded numpy
    arrays. The Python loop and dictionary have been replaced with a fully
    vectorized NumPy indexing approach for a massive speedup.
    """
    def __init__(self, read: pysam.AlignedSegment, seq_padding_token='-', quality_padding_value=np.nan):
        self.read = read
        # reference_start / reference_end can be None for some pathological reads;
        # handle defensively
        if self.read.reference_start is None or self.read.reference_end is None:
            # mark as zero-span read; callers should treat accordingly
            self.start = 0
            self.end = 0
        else:
            self.start = self.read.reference_start
            self.end = self.read.reference_end

        # total size of the covered region; if non-positive, create zero-length padded arrays
        span_length = max(0, self.end - self.start)

        # aligned positions that actually map to the reference (no soft-clips)
        aligned_positions_list = self.read.get_reference_positions(full_length=False)

        # create default padded arrays for all expected attributes
        self.padded_arrays = {
            'sequence': np.full(span_length, fill_value=seq_padding_token, dtype='<U1'),
            'base_qualities': np.full(span_length, fill_value=quality_padding_value, dtype=np.float32),
            'alignment_qualities': np.full(span_length, fill_value=quality_padding_value, dtype=np.float32)
        }

        # if no aligned positions (e.g., totally soft-clipped), return with padding arrays intact
        if not aligned_positions_list or span_length == 0:
            return

        # these are the aligned positions in absolute coordinates
        aligned_positions = np.array(aligned_positions_list, dtype=np.int64)
        # relative coordinates within span
        destination_indices = aligned_positions - self.start

        # obtain query-aligned sequence and qualities for aligned portion
        # pysam gives query_alignment_sequence (string) and query_alignment_qualities (list)
        query_seq = np.fromiter(self.read.query_alignment_sequence, dtype='<U1')
        query_b_qual = np.array(self.read.query_alignment_qualities, dtype=np.float32)

        # ensure sequence and qualities have the same length as destination_indices
        if len(query_seq) > len(destination_indices):
            query_seq = query_seq[:len(destination_indices)]
        if len(query_b_qual) > len(destination_indices):
            query_b_qual = query_b_qual[:len(destination_indices)]

        # place the read data into the padded arrays at the destination indices
        # (numpy will broadcast indices if shapes are correct)
        self.padded_arrays['sequence'][destination_indices] = query_seq
        self.padded_arrays['base_qualities'][destination_indices] = query_b_qual

        # For consistency with the original code's apparent intent.
        aligned_base_qualities = np.array(self.read.query_alignment_qualities, dtype=np.float32)
        if len(aligned_base_qualities) > len(destination_indices):
            aligned_base_qualities = aligned_base_qualities[:len(destination_indices)]
        self.padded_arrays['alignment_qualities'][destination_indices] = aligned_base_qualities


    def get_padded_array(self, attribute: str):
        """Retrieves a specific padded array by name."""
        if attribute not in self.padded_arrays:
            raise ValueError(f"Attribute '{attribute}' not available.")
        return self.padded_arrays[attribute]


class ReadFetcher:
    """
    IMPROVED: Assigns stable row_index to each MaskedRead so every encoder writes
    into the same row (the root cause of your mismatched padded regions).
    """
    def __init__(self, bam_file: str, max_reads: int, channel_encoders: list, genome: str, device='cpu'):
        if max_reads <= 0:
            raise ValueError("max_reads must be a positive integer.")

        self.bam_file = bam_file
        # Ensure BAM index present
        if not os.path.exists(bam_file + ".bai"):
            logging.info(f"Index for {bam_file} not found. Attempting to create one.")
            pysam.index(bam_file)
        self.bam = pysam.AlignmentFile(bam_file, "rb")

        self.max_reads = max_reads
        self.channel_encoders = channel_encoders
        self.genome = genome
        if not os.path.exists(genome + ".fai"):
            logging.info(f"Index file for genome {genome} not found. Creating index...")
            pysam.faidx(genome)
        self.device = device

    def fetch_tensors(self, chromosome: str, start: int, end: int, variant: VCFRegion) -> dict:
        region_obj = Region(f"{chromosome}:{start}-{end}")
        ref_seq = retrieve_sequence(self.genome, region_obj)

        # Fetch raw reads and pre-filter unmapped/secondary/supplementary/duplicates
        raw_iter = self.bam.fetch(chromosome, start, end)
        reads_kept = []
        for r in raw_iter:
            if r.is_unmapped or r.is_secondary or r.is_supplementary or r.is_duplicate:
                continue
            reads_kept.append(r)

        total_reads = len(reads_kept)
        if total_reads == 0:
            # No reads; create empty (filled) tensors from encoders and return
            region_length = end - start
            tensor_shape = (self.max_reads, region_length)
            tensors = {}
            for encoder in self.channel_encoders:
                tensors[encoder.name] = encoder(
                    reads_to_process=[],
                    start_row=(self.max_reads // 2),
                    tensor_shape=tensor_shape,
                    region_start=start,
                    device=self.device,
                    reference_sequence=ref_seq,
                    variant=variant
                )
            return tensors

        # Crop to the central max_reads if we have too many
        if total_reads > self.max_reads:
            logging.warning(
                f"Found {total_reads} reads, which is more than max_reads={self.max_reads}. "
                f"Taking the central {self.max_reads} reads."
            )
            crop_start = (total_reads - self.max_reads) // 2
            selected_reads = reads_kept[crop_start : crop_start + self.max_reads]
            start_row = 0
        else:
            selected_reads = reads_kept
            start_row = (self.max_reads - total_reads) // 2

        # Create MaskedRead objects in the selected order and assign stable row indices
        masked_reads = []
        for idx, r in enumerate(selected_reads):
            m = MaskedRead(r)
            # row_index is the canonical per-read index used by encoders
            m.row_index = idx
            masked_reads.append(m)

        region_length = end - start
        tensor_shape = (self.max_reads, region_length)
        tensors = {}
        # Pass the same reads_to_process (with row_index) to all encoders
        for encoder in self.channel_encoders:
            tensors[encoder.name] = encoder(
                reads_to_process=masked_reads,
                start_row=start_row,
                tensor_shape=tensor_shape,
                region_start=start,
                device=self.device,
                reference_sequence=ref_seq,
                variant=variant
            )

        return tensors

    def close(self):
        """Close the underlying BAM file gracefully."""
        if hasattr(self, 'bam') and self.bam:
            self.bam.close()

    def __del__(self):
        # Ensure file is closed
        try:
            self.close()
        except Exception:
            pass


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
