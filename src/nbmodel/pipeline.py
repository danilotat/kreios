import logging
from typing import Iterator, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd
import h5py
from scipy.stats import chi2_contingency
from reader import RibotishReader
from gtf import TranscriptCollector
from vcf import VariantCollector, VariantObj


@dataclass
class DatasetEntry:
    """A single entry ready for RiboseqDataset."""
    tid: str
    variant: VariantObj
    positions: 'np.ndarray'
    counts: 'np.ndarray'
    variant_idx: int


class DatasetBuilder:
    def __init__(
        self,
        variant_collector: VariantCollector,
        transcript_collector: TranscriptCollector,
        ribotish_reader: RibotishReader,
        consequence: str = 'frameshift',
    ):
        """
        Initialize the dataset builder.

        Args:
            variant_collector: VariantCollector with variants indexed by transcript
            transcript_collector: TranscriptCollector with transcript boundaries
            ribotish_reader: RibotishReader with ribosome profiles
            consequence: Variant consequence type to filter for (default: 'frameshift')
        """
        self.variants = variant_collector
        self.transcripts = transcript_collector
        self.profiles = ribotish_reader
        self.consequence = consequence
        self.profiles = self._build_dataframe()

    def _build_dataframe(self):
        results = defaultdict(list)
        for variant in self.variants:
            if any([self.consequence in x for x in variant.consequence.split('&')]):
                if variant.filtered:
                    if not variant.tid:
                        raise ValueError(f"Not valid tid in variant {variant}")
                    tid_start, tid_end, strand = self.transcripts[variant.tid]
                    if any([tid_start is None, tid_end is None]):
                        logging.warning(f"Unable to retrieve coordinates for transcript {variant.tid}")
                        continue

                    # Calculate relative position accounting for strand
                    # For forward (+) strand: variant_pos - tid_start
                    # For reverse (-) strand: tid_end - variant_pos (distance from 5' end)
                    if strand == '+':
                        rel_pos_variant = variant.pos - tid_start
                        transcript_len = tid_end - tid_start
                    else:  # reverse strand
                        rel_pos_variant = tid_end - variant.pos
                        transcript_len = tid_end - tid_start

                    _, pre_profile = self.profiles._get_profile(
                        variant.tid, 0, rel_pos_variant
                    )
                    _, after_profile = self.profiles._get_profile(
                        variant.tid, rel_pos_variant, transcript_len
                    )
                    if variant.tid in results:
                        logging.warning(f"{variant.tid} already present in the collector!!!")

                    features = self.transcripts.get_relative_features(variant.tid)
                    results[variant.tid].append(
                        {
                        'variant_id': variant.variant_id,
                        'variant_pos': variant.pos,
                        'variant_rel_pos': rel_pos_variant,
                        'strand': strand,
                        'pre_profile': pre_profile,
                        'after_profile': after_profile,
                        'features': features}
                    )
        return results

    def __repr__(self):
        return f"DatasetBuilder(consequence='{self.consequence}', n_variants={len(self.profiles)})"

    def to_hdf5(self, filepath: str) -> None:
        """
        Save the dataset to an HDF5 file.

        Structure:
            /<tid>/
                variant_<i>/
                    pre_profile: dataset (int array)
                    after_profile: dataset (int array)
                    features/
                        types: dataset (string array)
                        starts: dataset (int array)
                        ends: dataset (int array)
                    attrs:
                        variant_id: str
                        variant_pos: int
                        variant_rel_pos: int
                        strand: str

        Args:
            filepath: Path to output HDF5 file
        """
        with h5py.File(filepath, 'w') as f:
            for tid, entries in self.profiles.items():
                tid_grp = f.create_group(tid)

                for i, entry in enumerate(entries):
                    var_grp = tid_grp.create_group(f'variant_{i}')

                    # Store profiles as datasets
                    var_grp.create_dataset('pre_profile', data=entry['pre_profile'])
                    var_grp.create_dataset('after_profile', data=entry['after_profile'])

                    # Store features in a subgroup
                    features = entry['features']
                    if features:
                        feat_grp = var_grp.create_group('features')
                        feat_types = [f[0] for f in features]
                        feat_starts = [f[1] for f in features]
                        feat_ends = [f[2] for f in features]
                        # Use variable-length string dtype for feature types
                        dt = h5py.string_dtype(encoding='utf-8')
                        feat_grp.create_dataset('types', data=feat_types, dtype=dt)
                        feat_grp.create_dataset('starts', data=feat_starts)
                        feat_grp.create_dataset('ends', data=feat_ends)

                    # Store scalar metadata as attributes
                    var_grp.attrs['variant_id'] = entry['variant_id']
                    var_grp.attrs['variant_pos'] = entry['variant_pos']
                    var_grp.attrs['variant_rel_pos'] = entry['variant_rel_pos']
                    var_grp.attrs['strand'] = entry['strand']

        logging.info(f"Saved {len(self.profiles)} transcripts to {filepath}")

    @classmethod
    def from_hdf5(cls, filepath: str) -> dict:
        """
        Load dataset from an HDF5 file.

        Args:
            filepath: Path to HDF5 file

        Returns:
            Dict with same structure as self.profiles
        """
        profiles = {}
        with h5py.File(filepath, 'r') as f:
            for tid in f.keys():
                tid_grp = f[tid]
                entries = []

                for var_key in sorted(tid_grp.keys()):
                    var_grp = tid_grp[var_key]

                    entry = {
                        'variant_id': var_grp.attrs['variant_id'],
                        'variant_pos': int(var_grp.attrs['variant_pos']),
                        'variant_rel_pos': int(var_grp.attrs['variant_rel_pos']),
                        'strand': var_grp.attrs['strand'],
                        'pre_profile': var_grp['pre_profile'][:],
                        'after_profile': var_grp['after_profile'][:],
                    }

                    # Load features if present
                    if 'features' in var_grp:
                        feat_grp = var_grp['features']
                        types = [t.decode() if isinstance(t, bytes) else t for t in feat_grp['types'][:]]
                        starts = feat_grp['starts'][:]
                        ends = feat_grp['ends'][:]
                        entry['features'] = list(zip(types, starts, ends))
                    else:
                        entry['features'] = []

                    entries.append(entry)

                profiles[tid] = entries

        return profiles

    @staticmethod
    def _sum_by_frame(profile: np.ndarray) -> tuple[int, int, int]:
        """Sum counts by reading frame (0, 1, 2)."""
        frame_sums = [0, 0, 0]
        for i, count in enumerate(profile):
            frame_sums[i % 3] += int(count)
        return tuple(frame_sums)

    @staticmethod
    def _chi2_test_3x2(pre_profile: np.ndarray, after_profile: np.ndarray) -> dict:
        """
        Compute chi-square test on 3x2 contingency table (3 frames × pre/after).

        Returns:
            Dict with chi2 statistic, p-value, and degrees of freedom.
            Returns NaN values if test cannot be computed (e.g., zero counts).
        """
        pre_frames = DatasetBuilder._sum_by_frame(pre_profile)
        after_frames = DatasetBuilder._sum_by_frame(after_profile)

        # 3x2 contingency table: rows = frames, cols = pre/after
        table = np.array([
            [pre_frames[0], after_frames[0]],
            [pre_frames[1], after_frames[1]],
            [pre_frames[2], after_frames[2]],
        ])

        # Check for zero row/column sums which cause chi2_contingency to fail
        if table.sum() == 0 or (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
            return {'chi2': np.nan, 'p_value': np.nan, 'dof': np.nan}

        try:
            chi2, p_value, dof, _ = chi2_contingency(table)
            return {'chi2': chi2, 'p_value': p_value, 'dof': dof}
        except ValueError:
            return {'chi2': np.nan, 'p_value': np.nan, 'dof': np.nan}

    def to_pandas(self, with_chi2: bool = True) -> pd.DataFrame:
        """
        Export dataset to a pandas DataFrame with summed counts per frame.

        Args:
            with_chi2: If True, include chi-square test p-value

        Returns:
            DataFrame with columns: tid, variant_id, variant_pos, strand,
            pre_frame_0/1/2, after_frame_0/1/2, and optionally chi2_pvalue
        """
        rows = []
        for tid, entries in self.profiles.items():
            for entry in entries:
                pre_frames = self._sum_by_frame(entry['pre_profile'])
                after_frames = self._sum_by_frame(entry['after_profile'])

                row = {
                    'tid': tid,
                    'variant_id': entry['variant_id'],
                    'variant_pos': entry['variant_pos'],
                    'variant_rel_pos': entry['variant_rel_pos'],
                    'strand': entry['strand'],
                    'pre_frame_0': pre_frames[0],
                    'pre_frame_1': pre_frames[1],
                    'pre_frame_2': pre_frames[2],
                    'after_frame_0': after_frames[0],
                    'after_frame_1': after_frames[1],
                    'after_frame_2': after_frames[2],
                }

                if with_chi2:
                    chi2_result = self._chi2_test_3x2(
                        entry['pre_profile'], entry['after_profile']
                    )
                    row['chi2_pvalue'] = chi2_result['p_value']
                    row['chi2_statistic'] = chi2_result['chi2']

                rows.append(row)

        return pd.DataFrame(rows)




if __name__ == '__main__':
    from gtf import TranscriptCollector
    from vcf import VariantCollector
    from reader import RibotishReader

    # Example usage
    tc = TranscriptCollector(
        "/Users/danilo/Research/Tools/kreios/examples/ref/gencode.v48.annotation.gtf"
    )
    vc = VariantCollector(
        "/Users/danilo/Research/Tools/kreios/examples/vcf/SRR20649716_GSM6395076.deepvariant.phased.vep.vcf.gz"
    )
    rr = RibotishReader(
        "/Users/danilo/Research/Tools/kreios/examples/ribotish/"
        "SRR15513184_GSM5527709_Fibroblast_40_RiboSeq_Homo_sapiens_RNA-Seq_transprofile.py"
    )

    builder = DatasetBuilder(
        variant_collector=vc,
        transcript_collector=tc,
        ribotish_reader=rr,
        consequence='frameshift',
    )
    print(builder)

    # Export to pandas with chi-square test
    df = builder.to_pandas()
    print(df)

