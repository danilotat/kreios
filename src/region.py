#!/bin/python3

import cyvcf2
import bionumpy as bnp

class Region:
    def __init__(self, region: str):
        chromosome, positions = region.split(':')
        start, end = [int(float(j)) for j in positions.split('-')]
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
        self.identifier = f"{self.chromosome}:{self.start}:{self.ref}>{self.alt}"
    
    def __str__(self):
        return f"VCFRegion({self.identifier})"