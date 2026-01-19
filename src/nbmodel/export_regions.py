#!/usr/bin/env python3
import argparse
import pandas as pd
from .vcf import VariantCollector
from .alignment import RegionCollector


def main():
    parser = argparse.ArgumentParser(description="Export RegionCollector regions to CSV")
    parser.add_argument("--vcf", required=True, help="Input VCF file")
    parser.add_argument("--bam", required=True, help="Input BAM file")
    parser.add_argument("--outfile", required=True, help="Output CSV file")
    args = parser.parse_args()

    vc = VariantCollector(args.vcf)
    rc = RegionCollector(args.bam, vc)

    df = pd.DataFrame.from_dict(rc._regions, orient='index')
    df.index.name = 'variant_id'
    df.to_csv(args.outfile)


if __name__ == '__main__':
    main()
