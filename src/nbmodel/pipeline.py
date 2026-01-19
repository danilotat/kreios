import logging
from typing import Iterator, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import h5py
from reader import RibotishReader, RibotishORFReader, ORFRecord
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
        orf_reader: RibotishORFReader,
        consequence: str = 'frameshift',
    ):
        """
        Initialize the dataset builder.

        Args:
            variant_collector: VariantCollector with variants indexed by transcript
            transcript_collector: TranscriptCollector with transcript boundaries
            ribotish_reader: RibotishReader with ribosome profiles
            orf_reader: RibotishORFReader with ORF predictions
            consequence: Variant consequence type to filter for (default: 'frameshift')
        """
        self.variants = variant_collector
        self.transcripts = transcript_collector
        self.profiles = ribotish_reader
        self.orfs = orf_reader
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




if __name__ == '__main__':
    from gtf import TranscriptCollector
    from vcf import VariantCollector
    from reader import RibotishReader, RibotishORFReader

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
    orfs = RibotishORFReader(
        "/Users/danilo/Research/Tools/kreios/examples/ribotish/"
        "SRR15513184_GSM5527709_Fibroblast_40_RiboSeq_Homo_sapiens_RNA-Seq_pred.txt"
    )


    builder = DatasetBuilder(
        variant_collector=vc,
        transcript_collector=tc,
        ribotish_reader=rr,
        orf_reader=orfs,
        consequence='frameshift',  # default, can be changed to other consequences
    )
    print(builder)
    for tid, profiles in builder.profiles.items():
        for i in profiles:
            pre_sum = i['pre_profile'].sum()
            after_sum = i['after_profile'].sum()
            features = i.get('features', [])
            print(f'Tid: {tid}\tpre: {pre_sum}\tafter: {after_sum}\tfeatures: {features}')

    print(builder.profiles.get('ENST00000381605'))
    
