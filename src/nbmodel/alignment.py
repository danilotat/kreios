import pysam 
from vcf import VariantCollector
import numpy as np
import scipy.stats as stats

class CoverageHandler:
    def __init__(self, coverage: np.ndarray):
        self.cov = dict(zip(['A','C','G','T'], coverage))
        self.dp = np.sum(coverage)
    
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
            if variant.is_snp:
                cov = CoverageHandler(self._bam_file.count_coverage(
                    variant.chrom, variant.pos, variant.pos + 1,
                    read_callback="BAM_FQCFAIL"
                ))
                binom_p = stats.binomtest(
                    cov.get(variant.alt),
                    cov.dp,
                    p=np.round(
                        variant.alt_dp / variant.dp, 3
                    )
                ).pvalue
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
    



