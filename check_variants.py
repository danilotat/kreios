from src.pileup import PileupRegion
from src.region import VCFRegion
import logging
import cyvcf2
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stderr)
    ])

if __name__ == "__main__":
    vcf_file = sys.argv[1]
    bam_file = sys.argv[2]
    vcf = cyvcf2.VCF(vcf_file)
    j = 0
    for variant in vcf:
        if j > 2000:
            break
        else:
            if variant.is_snp:
                region = VCFRegion(variant.CHROM, variant.POS, variant.POS+1, variant)
                pileup_region = PileupRegion(region.chromosome, region.start, region.end, bam_file)
                depth = pileup_region.compute_depth(region)
                if depth.tot == 0:
                    continue
                    # logging.warning(f"No reads at {region}")
                else:
                    if depth.tot < 10:     
                        logging.warning(f"Low depth ({depth.tot}) at {region}")
                    else:
                        print(f"Variant: {region}, Depth: {depth}")
                        j += 1
            