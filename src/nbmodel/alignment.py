import pysam
from vcf import VariantCollector
from gtf import TranscriptCollector
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Optional, Any
import logging

class CoverageHandler:
    def __init__(self, coverage: np.ndarray):
        self.cov = dict(zip(['A','C','G','T'], [c[0] if c[0] else 0 for c in coverage]))
        self.dp = int(np.sum(coverage))
    
    def get(self, n: str):
        try:
            return self.cov[n]
        except KeyError:
            return 0
    
class RegionCollector:
    def __init__(self, bam_file: str, vc: VariantCollector):
        self._bam_file = pysam.AlignmentFile(bam_file, 'rb')
        self._regions = self._store_regions(vc)
    
    def _store_regions(self, vc: VariantCollector) -> dict:
        collector = {}
        for variant in vc:
            if all([
                variant.is_snp,
                'missense' in variant.consequence,
                variant.filtered]
                ):
                cov = CoverageHandler(self._bam_file.count_coverage(
                    variant.chrom, variant.pos, variant.pos + 1,
                    read_callback="all"
                ))
                # binom test requires n >= 1 and valid k
                if cov.dp >= 1 and cov.get(variant.alt) >= 0:
                    binom_p = stats.binomtest(
                        cov.get(variant.alt),
                        cov.dp,
                        p=np.round(
                            variant.alt_dp / variant.dp, 3
                        )
                    ).pvalue
                else:
                    binom_p = None
                collector[variant.variant_id] = {
                    'tid': variant.tid,
                    'dp_RNA': variant.dp,
                    'alt_dp_RNA': variant.alt_dp,
                    'dp_RIBO': cov.dp,
                    'alt_dp_RIBO': cov.get(variant.alt),
                    'p': binom_p
                }
        return collector

    def get(self, variantid: str):
        try:
            return self._regions[variantid]
        except KeyError:
            return {}


class VariantCDSCoverage:
    """
    Collects RNA coverage from variant position to CDS end for variants
    matching a given consequence type.
    """

    def __init__(
        self,
        bam_file: str,
        variant_collector: VariantCollector,
        transcript_collector: TranscriptCollector,
        consequence: str = 'frameshift'
    ):
        self._bam = pysam.AlignmentFile(bam_file, 'rb')
        self._variants = variant_collector
        self._transcripts = transcript_collector
        self._consequence = consequence
        self._data: Dict[str, Dict[str, Any]] = {}
        self._collect()

    def _get_cds_bounds(self, tid: str) -> Optional[tuple]:
        """
        Get the genomic CDS start and end for a transcript.
        Returns (cds_start, cds_end) in genomic coordinates, or None if not found.
        """
        if tid not in self._transcripts.transcripts:
            return None

        features = self._transcripts.transcripts[tid].get('features', [])
        cds_regions = [(start, end) for feat_type, start, end in features if feat_type == 'CDS']

        if not cds_regions:
            return None

        cds_start = min(start for start, end in cds_regions)
        cds_end = max(end for start, end in cds_regions)
        return (cds_start, cds_end)

    def _get_coverage_array(self, chrom: str, start: int, end: int) -> np.ndarray:
        """
        Get total coverage for a genomic region using pysam.
        Returns a numpy array of coverage values for each position.
        """
        if start >= end:
            return np.array([])

        coverage = self._bam.count_coverage(chrom, start, end, read_callback="all")
        return np.sum(coverage, axis=0)

    def _collect(self):
        """
        Process all variants matching the consequence type and collect
        coverage from variant position to CDS end.
        """
        for variant in self._variants:
            if not any(self._consequence in c for c in variant.consequence.split('&')):
                continue

            if not variant.filtered:
                continue

            tid = variant.tid
            tx_start, tx_end, strand = self._transcripts[tid]

            if tx_start is None:
                logging.warning(f"Transcript {tid} not found for variant {variant.variant_id}")
                continue

            cds_bounds = self._get_cds_bounds(tid)
            if cds_bounds is None:
                logging.warning(f"No CDS found for transcript {tid}")
                continue

            cds_start, cds_end = cds_bounds

            # Determine coverage region from variant to CDS end based on strand
            if strand == '+':
                # Forward strand: variant pos to CDS end
                cov_start = variant.pos
                cov_end = cds_end
                # Relative position within CDS
                variant_cds_pos = variant.pos - cds_start
            else:
                # Reverse strand: CDS start to variant pos
                cov_start = cds_start
                cov_end = variant.pos + 1
                # Relative position within CDS (from 5' end of CDS)
                variant_cds_pos = cds_end - variant.pos - 1

            coverage = self._get_coverage_array(variant.chrom, cov_start, cov_end)

            # For reverse strand, flip coverage to be 5' to 3'
            if strand == '-':
                coverage = coverage[::-1]

            self._data[variant.variant_id] = {
                'tid': tid,
                'strand': strand,
                'variant_pos': variant.pos,
                'cds_start': cds_start,
                'cds_end': cds_end,
                'variant_cds_pos': variant_cds_pos,
                'coverage': coverage,
                'consequence': variant.consequence
            }

    def get(self, variant_id: str) -> Optional[Dict[str, Any]]:
        """Get coverage data for a specific variant."""
        return self._data.get(variant_id)

    def __getitem__(self, variant_id: str) -> Dict[str, Any]:
        if variant_id not in self._data:
            raise KeyError(f"Variant {variant_id} not found")
        return self._data[variant_id]

    def __iter__(self):
        return iter(self._data.items())

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"VariantCDSCoverage with {len(self._data)} variants"
