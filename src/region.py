#!/bin/python3

import cyvcf2
import bionumpy as bnp

class Region:
    def __init__(self, region: str):
        chromosome, positions = region.split(':')
        start, end = map(int, positions.split('-'))
        self.chromosome = chromosome
        self.start = start
        self.end = end
        self.length = end - start + 1
    
    def _to_bnp_interval(self):
        interval = bnp.datatypes.Interval(
            [self.chromosome],
            [self.start],
            [self.end])
        return interval

    def __str__(self):
        return f"{self.chromosome}:{self.start}-{self.end}"
    
    def __repr__(self):
        return f"Region({self.chromosome}, {self.start}, {self.end})"


class VCFRegion(Region):
    def __init__(self, region, ref: str, alt: str):
        super().__init__(region)
        self.ref = ref
        self.alt = alt
    
    def __str__(self):
        return f"VCFRegion({self.chromosome}:{self.start}:{self.ref}>{self.alt})"


# class VCFRegion(Region):
#     def __init__(self, region: str, variant: cyvcf2.Variant):
#         super().__init__(region)
#         self.variant = variant
#         self.ref, self.alt = variant.REF, variant.ALT[0]
#         self.EVFS = variant.INFO.get('EVFS', None)
#         self.DP = variant.gt_depths[0]
#         self.ref_DP, self.alt_DP = variant.gt_ref_depths[0], variant.gt_alt_depths[0]
#         self.VAF = variant.gt_alt_freqs[0]
    
#     def __str__(self):
#         return f"{self.chromosome}:{self.start}-{self.end} {self.ref}>{self.alt} VAF={self.VAF} DP={self.DP}"
    
#     def __repr__(self):
#         return f"VCFRegion({self.chromosome}, {self.start}, {self.end}, {self.ref}, {self.alt}, VAF={self.VAF}, DP={self.DP})"