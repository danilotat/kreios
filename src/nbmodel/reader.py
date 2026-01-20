import re
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Iterator, Union, Optional
from intervaltree import IntervalTree
from pathlib import Path

_RIBOTISH_COLS = ["Gid","Tid","Symbol","RiboProf"]
_RIBOTISH_ORF_COLS = ["Gid", "Tid", "Symbol", "GenomePos", "Start", "Stop",
                      "StartCodon", "TisType", "AALen"]
_GENOME_POS_PATTERN = re.compile(r'^(.+):(\d+)-(\d+):([+-])$')


@dataclass(slots=True)
class ORFRecord:
    """Single ORF prediction record (compatible with ribotish)."""
    gid: str
    tid: str
    symbol: str
    chromosome: str
    genome_start: int
    genome_end: int
    strand: str
    tx_start: int
    tx_stop: int
    start_codon: str
    tis_type: str
    aa_len: int


class RiboWaltzReader:
    """
    Reader for riboWaltz P-site output files (*.psite.tsv.gz).
    
    Provides interface compatible with RibotishReader for downstream analysis.
    riboWaltz outputs per-read P-site positions which are aggregated to 
    nucleotide-level counts for profile analysis.
    
    Expected columns:
        transcript, end5, psite, end3, length, cds_start, cds_stop,
        psite_from_start, psite_from_stop, psite_region, p_site_codon, sample
    
    Example:
        reader = RiboWaltzReader("Fibroblast_45.psite.tsv.gz")
        
        # Get profile for a region
        positions, counts = reader._get_profile("ENST00000428771", 100, 500)
        
        # Get frame-split profile for an ORF
        frames = reader.get_frame_profile("ENST00000428771", orf)
        
        # Get dataset entry compatible with RiboseqDataset
        entry = reader.get_dataset_entry("ENST00000428771", orf, variant_idx=10)
    """
    
    # Expected columns in riboWaltz psite output
    _REQUIRED_COLS = ['transcript', 'psite']
    _OPTIONAL_COLS = ['end5', 'end3', 'length', 'cds_start', 'cds_stop',
                      'psite_from_start', 'psite_from_stop', 'psite_region',
                      'p_site_codon', 'sample']
    
    def __init__(self, file: Union[str, Path], 
                 strip_version: bool = True,
                 keep_raw: bool = False):
        """
        Initialize reader and load P-site data.
        
        Args:
            file: Path to riboWaltz psite.tsv.gz file
            strip_version: Whether to strip version from transcript IDs (default: True)
            keep_raw: Whether to keep raw per-read data in memory (default: False)
        """
        self._file = Path(file)
        self._strip_version = strip_version
        self._sample_name: Optional[str] = None
        self._n_reads: int = 0
        
        # Load and process data
        self._load_and_aggregate(keep_raw)
    
    def _load_and_aggregate(self, keep_raw: bool) -> None:
        """Load psite file and aggregate to position-level counts."""
        # Read full file
        df = pd.read_csv(self._file, sep='\t', compression='gzip')
        
        # Validate required columns
        missing = set(self._REQUIRED_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self._n_reads = len(df)
        
        # Extract sample name if available
        if 'sample' in df.columns and len(df) > 0:
            self._sample_name = df['sample'].iloc[0]
        
        # Strip version from transcript IDs
        if self._strip_version:
            df['transcript'] = df['transcript'].str.split('.').str[0]
        
        # Store raw data if requested
        if keep_raw:
            self._raw_frame = df.copy()
        
        # Aggregate to position-level counts
        # Group by transcript and psite, count occurrences
        self._aggregated = (
            df.groupby(['transcript', 'psite'])
            .size()
            .reset_index(name='count')
        )
        
        # Build index for fast transcript lookup
        # Store as dict: tid -> DataFrame slice
        self._by_transcript: dict[str, pd.DataFrame] = {}
        for tid, group in self._aggregated.groupby('transcript'):
            # Convert to numpy for faster access
            self._by_transcript[tid] = {
                'positions': group['psite'].values,
                'counts': group['count'].values
            }
    
    @staticmethod
    def _strip_version_id(identifier: str) -> str:
        """Strip version number from transcript/gene ID."""
        return identifier.split('.')[0]
    
    def _get_profile(self, tid: str, p_from: int, p_to: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Get P-site profile for a transcript region.
        
        Args:
            tid: Transcript ID (version will be stripped if strip_version=True)
            p_from: Start position (inclusive, 0-based)
            p_to: End position (inclusive)
            
        Returns:
            Tuple of (positions, counts) arrays covering the full range
        """
        if self._strip_version:
            tid = self._strip_version_id(tid)
        
        # Get pre-indexed data for this transcript
        tx_data = self._by_transcript.get(tid)
        
        if tx_data is None:
            logging.warning(
                f"Transcript {tid} not found in {self._file.name}. "
                "Returning empty arrays."
            )
            return np.array([], dtype=int), np.array([], dtype=int)
        
        positions = tx_data['positions']
        counts = tx_data['counts']
        
        # Create output arrays for requested range (dense)
        out_positions = np.arange(p_from, p_to + 1)
        out_counts = np.zeros(len(out_positions), dtype=int)
        
        # Filter to requested range
        mask = (positions >= p_from) & (positions <= p_to)
        valid_positions = positions[mask]
        valid_counts = counts[mask]
        
        # Map sparse positions to dense output
        if len(valid_positions) > 0:
            indices = valid_positions - p_from
            out_counts[indices] = valid_counts
        
        return out_positions, out_counts
    
    def get_sparse_profile(self, tid: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Get sparse P-site profile for entire transcript.
        
        More memory efficient than dense profile for transcripts with
        sparse coverage.
        
        Args:
            tid: Transcript ID
            
        Returns:
            Tuple of (positions, counts) arrays (sparse, only non-zero)
        """
        if self._strip_version:
            tid = self._strip_version_id(tid)
        
        tx_data = self._by_transcript.get(tid)
        
        if tx_data is None:
            logging.warning(f"Transcript {tid} not found.")
            return np.array([], dtype=int), np.array([], dtype=int)
        
        return tx_data['positions'].copy(), tx_data['counts'].copy()
    
    def get_frame_profile(self, tid: str, orf: 'ORFRecord') -> np.ndarray:
        """
        Get ribosome profile split by reading frame for an ORF.
        
        Args:
            tid: Transcript ID
            orf: ORFRecord containing tx_start and tx_stop
            
        Returns:
            Array of shape (n_codons, 3) where columns are frame 0, 1, 2
        """
        positions, profile = self._get_profile(tid, orf.tx_start, orf.tx_stop)
        
        if len(positions) == 0:
            return np.zeros((0, 3), dtype=int)
        
        # Calculate number of complete codons
        n_positions = len(positions)
        n_codons = n_positions // 3
        
        if n_codons == 0:
            return np.zeros((0, 3), dtype=int)
        
        # Vectorized frame assignment
        frames = np.zeros((n_codons, 3), dtype=int)
        
        # Calculate frame and codon index for each position
        rel_positions = positions[:n_codons * 3] - orf.tx_start
        frame_indices = rel_positions % 3
        codon_indices = rel_positions // 3
        
        # Use advanced indexing to fill frames array
        # This is faster than a loop for large arrays
        for i, (codon_idx, frame, count) in enumerate(
            zip(codon_indices, frame_indices, profile[:n_codons * 3])
        ):
            if codon_idx < n_codons:
                frames[codon_idx, frame] = count
        
        return frames
    
    def get_transcript_frames(self, tid: str, orf_reader: 'RibotishORFReader',
                              orf_index: int = 0) -> np.ndarray:
        """
        Get frame profile for a transcript using ORF predictions.
        
        Args:
            tid: Transcript ID
            orf_reader: RibotishORFReader instance
            orf_index: Which ORF to use if multiple (default: first/annotated)
            
        Returns:
            Array of shape (n_codons, 3)
        """
        orfs = orf_reader.get_by_transcript(tid)
        if not orfs:
            logging.warning(f"No ORFs found for {tid}")
            return np.zeros((0, 3), dtype=int)
        
        if orf_index >= len(orfs):
            logging.warning(
                f"ORF index {orf_index} out of range for {tid} "
                f"(has {len(orfs)} ORFs). Using index 0."
            )
            orf_index = 0
        
        orf = orfs[orf_index]
        return self.get_frame_profile(tid, orf)
    
    def get_dataset_entry(self, tid: str, orf: 'ORFRecord',
                          variant_idx: int = 0) -> dict:
        """
        Get data formatted for RiboseqDataset.
        
        Aggregates nucleotide-level profile into codon-level counts by frame.
        
        Args:
            tid: Transcript ID
            orf: ORFRecord containing tx_start and tx_stop
            variant_idx: Position of variant (in codon coordinates)
            
        Returns:
            Dict with:
                - 'positions': array of codon indices (n_codons,)
                - 'counts': array of shape (n_codons, 3) for frames 0, 1, 2
                - 'variant_idx': variant position
        """
        counts = self.get_frame_profile(tid, orf)
        n_codons = counts.shape[0]
        positions = np.arange(n_codons)
        
        return {
            'positions': positions,
            'counts': counts,
            'variant_idx': variant_idx
        }
    
    def get_coverage_stats(self, tid: str) -> dict:
        """
        Get coverage statistics for a transcript.
        
        Args:
            tid: Transcript ID
            
        Returns:
            Dict with coverage statistics
        """
        if self._strip_version:
            tid = self._strip_version_id(tid)
        
        tx_data = self._by_transcript.get(tid)
        
        if tx_data is None:
            return {
                'total_reads': 0,
                'covered_positions': 0,
                'max_coverage': 0,
                'mean_coverage': 0.0
            }
        
        counts = tx_data['counts']
        return {
            'total_reads': int(counts.sum()),
            'covered_positions': len(counts),
            'max_coverage': int(counts.max()),
            'mean_coverage': float(counts.mean())
        }
    
    @property
    def sample_name(self) -> Optional[str]:
        """Sample name from the file."""
        return self._sample_name
    
    @property
    def transcripts(self) -> list[str]:
        """List of transcript IDs with P-site data."""
        return list(self._by_transcript.keys())
    
    @property
    def n_reads(self) -> int:
        """Total number of reads in the file."""
        return self._n_reads
    
    @property
    def n_transcripts(self) -> int:
        """Number of transcripts with P-site data."""
        return len(self._by_transcript)
    
    @property
    def total_psites(self) -> int:
        """Total P-site count across all positions."""
        return sum(
            data['counts'].sum() 
            for data in self._by_transcript.values()
        )
    
    def __repr__(self) -> str:
        return (
            f"RiboWaltzReader({self._file.name}, "
            f"sample={self._sample_name}, "
            f"{self.n_reads:,} reads, "
            f"{self.n_transcripts:,} transcripts)"
        )
    
    def __contains__(self, tid: str) -> bool:
        """Check if transcript has P-site data."""
        if self._strip_version:
            tid = self._strip_version_id(tid)
        return tid in self._by_transcript

class RibotishReader(object):
    def __init__(self, file: str):
        self._file = file
        self._raw_frame = pd.read_csv(
            self._file, sep='\t', usecols=_RIBOTISH_COLS
        )
        self._raw_frame[['pos', 'profile']] = self._raw_frame['RiboProf'].apply(
            lambda rp: self.parse_riboprof(rp)
        ).apply(pd.Series)
        self._raw_frame.drop(columns=['RiboProf'], inplace=True)
        self._raw_frame.Tid = self._raw_frame.Tid.apply(lambda x: x.split('.')[0])
        self._raw_frame.Gid = self._raw_frame.Gid.apply(lambda x: x.split('.')[0]) 

    @staticmethod
    def parse_riboprof(rp):
        rp = rp.strip('{}')
        if not rp:
            return np.array([], dtype=int), np.array([], dtype=int)
        # Convert "77:1, 78:2" → "77,1,78,2" and parse directly
        cleaned = rp.replace(':', ',').replace(' ', '')
        data = np.fromstring(cleaned, sep=',', dtype=int)
        positions = data[::2]
        profile = data[1::2]
        return positions, profile

    def _get_profile(self, tid: str, p_from: int, p_to: int) -> tuple[np.ndarray, np.ndarray]:
        row = self._raw_frame.loc[self._raw_frame['Tid'] == tid]
        if len(row) == 0:
            logging.warning(f"The transcript {tid} was not found in {self._file}. Returning empty arrays")
            return np.array([]), np.array([])
        positions = row['pos'].iloc[0]
        profile = row['profile'].iloc[0]
        # Create output arrays for the requested range
        _positions = np.arange(p_from, p_to + 1)
        _profile = np.zeros(len(_positions), dtype=int)
        # Find which sparse positions fall within our range
        mask = (positions >= p_from) & (positions <= p_to)
        valid_positions = positions[mask]
        valid_profile = profile[mask]
        # Map sparse positions to indices in our output array
        indices = valid_positions - p_from
        _profile[indices] = valid_profile
        return _positions, _profile

    def get_frame_profile(self, tid: str, orf: 'ORFRecord') -> np.ndarray:
        """
        Get ribosome profile split by reading frame for an ORF.

        Args:
            tid: Transcript ID
            orf: ORFRecord containing tx_start and tx_stop

        Returns:
            Array of shape (n_codons, 3) where columns are frame 0, 1, 2
        """
        positions, profile = self._get_profile(tid, orf.tx_start, orf.tx_stop)

        if len(positions) == 0:
            return np.zeros((0, 3), dtype=int)

        # Calculate number of complete codons
        n_positions = len(positions)
        n_codons = n_positions // 3

        # Initialize frame array
        frames = np.zeros((n_codons, 3), dtype=int)

        # Split by frame: position relative to start codon
        for pos, count in zip(positions, profile):
            frame = (pos - orf.tx_start) % 3
            codon_idx = (pos - orf.tx_start) // 3
            if codon_idx < n_codons:
                frames[codon_idx, frame] = count

        return frames

    def get_transcript_frames(self, tid: str, orf_reader: 'RibotishORFReader',
                              orf_index: int = 0) -> np.ndarray:
        """
        Get frame profile for a transcript using ORF predictions.

        Args:
            tid: Transcript ID
            orf_reader: RibotishORFReader instance
            orf_index: Which ORF to use if multiple (default: first/annotated)

        Returns:
            Array of shape (n_codons, 3)
        """
        orfs = orf_reader.get_by_transcript(tid)
        if not orfs:
            logging.warning(f"No ORFs found for {tid}")
            return np.zeros((0, 3), dtype=int)

        orf = orfs[orf_index]
        return self.get_frame_profile(tid, orf)

    def get_dataset_entry(self, tid: str, orf: 'ORFRecord',
                          variant_idx: int = 0) -> dict:
        """
        Get data formatted for RiboseqDataset.

        Aggregates nucleotide-level profile into codon-level counts by frame.

        Args:
            tid: Transcript ID
            orf: ORFRecord containing tx_start and tx_stop
            variant_idx: Position of variant (in codon coordinates)

        Returns:
            Dict with:
                - 'positions': array of codon indices (n_codons,)
                - 'counts': array of shape (n_codons, 3) for frames 0, 1, 2
                - 'variant_idx': variant position
        """
        counts = self.get_frame_profile(tid, orf)
        n_codons = counts.shape[0]
        positions = np.arange(n_codons)

        return {
            'positions': positions,
            'counts': counts,
            'variant_idx': variant_idx
        }

    def __repr__(self):
        print(f"Ribotish reads from {self._file}")


class RibotishORFReader:
    """
    Reader for ribotish ORF prediction files.

    Provides efficient lookup of ORFs by:
    - Transcript ID: get_by_transcript(tid)
    - Genomic region: query_region(chrom, start, end)

    Example:
        orfs = RibotishORFReader("pred.txt")

        # Get all ORFs for a transcript
        transcript_orfs = orfs.get_by_transcript("ENST00000428771")

        # Query ORFs overlapping a genomic region
        overlapping = orfs.query_region("chr1", 999000, 1000000)
    """

    def __init__(self, file: str):
        """
        Initialize reader and parse ORF predictions.

        Args:
            file: Path to ribotish prediction TSV file
        """
        self._file = file
        self._orfs: list[ORFRecord] = []
        self._by_transcript: dict[str, list[int]] = {}
        self._by_chromosome: dict[str, IntervalTree] = {}

        self._load()

    @staticmethod
    def parse_genome_pos(genome_pos: str) -> tuple[str, int, int, str]:
        """
        Parse GenomePos string into components.

        Args:
            genome_pos: String like "chr1:999058-999973:-"

        Returns:
            Tuple of (chromosome, start, end, strand)
            Start is converted to 0-based for consistency.
        """
        match = _GENOME_POS_PATTERN.match(genome_pos)
        if not match:
            raise ValueError(f"Invalid GenomePos format: {genome_pos}")
        chrom, start, end, strand = match.groups()
        # ribotish uses 0-based half-open coordinates
        return chrom, int(start), int(end), strand

    @staticmethod
    def _strip_version(identifier: str) -> str:
        """Strip version number from gene/transcript ID."""
        return identifier.split('.')[0]

    def _load(self) -> None:
        """Load and index ORF predictions."""
        df = pd.read_csv(self._file, sep='\t', usecols=_RIBOTISH_ORF_COLS)

        # Strip versions vectorized
        df['Gid'] = df['Gid'].str.split('.').str[0]
        df['Tid'] = df['Tid'].str.split('.').str[0]

        # Parse GenomePos vectorized
        genome_parts = df['GenomePos'].str.extract(_GENOME_POS_PATTERN)
        df['chromosome'] = genome_parts[0]
        df['genome_start'] = genome_parts[1].astype(int)
        df['genome_end'] = genome_parts[2].astype(int)
        df['strand'] = genome_parts[3]

        # Build records and indices
        for idx, row in df.iterrows():
            orf = ORFRecord(
                gid=row['Gid'],
                tid=row['Tid'],
                symbol=row['Symbol'],
                chromosome=row['chromosome'],
                genome_start=row['genome_start'],
                genome_end=row['genome_end'],
                strand=row['strand'],
                tx_start=int(row['Start']),
                tx_stop=int(row['Stop']),
                start_codon=row['StartCodon'],
                tis_type=row['TisType'],
                aa_len=int(row['AALen'])
            )

            orf_idx = len(self._orfs)
            self._orfs.append(orf)

            # Index by transcript
            if orf.tid not in self._by_transcript:
                self._by_transcript[orf.tid] = []
            self._by_transcript[orf.tid].append(orf_idx)

            # Index by genomic position
            if orf.chromosome not in self._by_chromosome:
                self._by_chromosome[orf.chromosome] = IntervalTree()
            self._by_chromosome[orf.chromosome].addi(
                orf.genome_start, orf.genome_end, orf_idx
            )

    def get_by_transcript(self, tid: str) -> list[ORFRecord]:
        """
        Get all ORFs for a transcript.

        Args:
            tid: Transcript ID (with or without version)

        Returns:
            List of ORFRecord objects for this transcript
        """
        tid = self._strip_version(tid)
        indices = self._by_transcript.get(tid, [])
        return [self._orfs[i] for i in indices]

    def query_region(self, chrom: str, start: int, end: int) -> list[ORFRecord]:
        """
        Query ORFs overlapping a genomic region.

        Args:
            chrom: Chromosome name (e.g., "chr1")
            start: Start position (0-based)
            end: End position (exclusive)

        Returns:
            List of ORFRecord objects overlapping the region
        """
        tree = self._by_chromosome.get(chrom)
        if tree is None:
            return []

        intervals = tree.overlap(start, end)
        return [self._orfs[iv.data] for iv in intervals]

    def query_point(self, chrom: str, position: int) -> list[ORFRecord]:
        """
        Query ORFs containing a specific genomic position.

        Args:
            chrom: Chromosome name
            position: Genomic position (0-based)

        Returns:
            List of ORFRecord objects containing this position
        """
        return self.query_region(chrom, position, position + 1)

    @property
    def chromosomes(self) -> list[str]:
        """List of chromosomes with ORF predictions."""
        return list(self._by_chromosome.keys())

    @property
    def transcript_ids(self) -> list[str]:
        """List of transcript IDs with ORF predictions."""
        return list(self._by_transcript.keys())

    def __len__(self) -> int:
        """Total number of ORF predictions."""
        return len(self._orfs)

    def __repr__(self) -> str:
        return (f"RibotishORFReader({len(self)} ORFs, "
                f"{len(self._by_transcript)} transcripts, "
                f"{len(self._by_chromosome)} chromosomes)")

    def __iter__(self) -> Iterator[ORFRecord]:
        """Iterate over all ORF records."""
        return iter(self._orfs)


if __name__ == '__main__':
    # Test RibotishReader (profile data)
    rr = RibotishReader(
        "/Users/danilo/Research/Tools/kreios/examples/ribotish/"
        "SRR15513184_GSM5527709_Fibroblast_40_RiboSeq_Homo_sapiens_RNA-Seq_transprofile.py"
    )
    print("Profile test:", rr._get_profile('ENST00000514057', 0, 200))

    # Test RibotishORFReader (ORF predictions)
    orfs = RibotishORFReader(
        "/Users/danilo/Research/Tools/kreios/examples/ribotish/"
        "SRR15513184_GSM5527709_Fibroblast_40_RiboSeq_Homo_sapiens_RNA-Seq_pred.txt"
    )
    print(orfs)

    # Query by transcript
    hes4_orfs = orfs.get_by_transcript("ENST00000428771")
    print(f"\nHES4 transcript has {len(hes4_orfs)} ORFs")
    for orf in hes4_orfs[:3]:
        print(f"  {orf.tis_type}: {orf.start_codon} at tx:{orf.tx_start}, AALen={orf.aa_len}")

    # Query by genomic region
    chr1_region = orfs.query_region("chr1", 999000, 1000000)
    print(f"\nORFs in chr1:999000-1000000: {len(chr1_region)}")

    # Test frame profile methods
    if hes4_orfs:
        frames = rr.get_frame_profile('ENST00000428771', hes4_orfs[0])
        print(f"\nFrame profile shape: {frames.shape}")
        print(f"Frame sums (f0, f1, f2): {frames.sum(axis=0)}")
        print(f"Total reads: {frames.sum()}")

        # Test convenience method
        frames2 = rr.get_transcript_frames('ENST00000428771', orfs)
        print(f"Convenience method matches: {np.array_equal(frames, frames2)}")

        # Test dataset entry format
        entry = rr.get_dataset_entry('ENST00000428771', hes4_orfs[0], variant_idx=10)
        print(f"\nDataset entry:")
        print(f"  positions shape: {entry['positions'].shape}")
        print(f"  counts shape: {entry['counts'].shape}")
        print(f"  variant_idx: {entry['variant_idx']}")
        print(f"  First 3 codons: positions={entry['positions'][:3]}, counts=\n{entry['counts'][:3]}")