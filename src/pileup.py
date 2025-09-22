import pysam
import logging
from .region import Region
from collections import namedtuple
import sys

nt_coverage = namedtuple('coverage', ['A', 'C', 'G', 'T', 'tot'])

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ])

class PileupRegion(Region):
    def __init__(self, chromosome, start, end, bam_file):
        super().__init__(chromosome, start, end)
        self.bam_file = bam_file
    
    def compute_depth(self, region: Region) -> nt_coverage:
        """
        Compute the depth of coverage for the given region using pysam.
        """
        if region.length > 2:
            raise NotImplementedError("Depth computation just for SNPs")
        bam = pysam.AlignmentFile(self.bam_file, "rb")
        #pysam uses 0-based coordinates
        coverage = bam.count_coverage(region.chromosome, region.start, region.end)
        nt_cov = nt_coverage(*[sum(cov) for cov in coverage] + [sum([sum(cov) for cov in coverage])])
        bam.close()
        return nt_cov 



