#!/usr/bin/env python3
import argparse
from vcf import VariantCollector
from gtf import TranscriptCollector
from reader import RibotishReader, RiboWaltzReader
from pipeline import DatasetBuilder


def main():
    parser = argparse.ArgumentParser(
        description="Export frameshift variant profiles to CSV with chi-square test"
    )
    parser.add_argument("--vcf", required=True, help="Input VCF file with VEP annotations")
    parser.add_argument("--gtf", required=True, help="GTF annotation file")
    parser.add_argument("--profile", required=True, help="Ribosome profile file (ribotish transprofile or riboWaltz psite)")
    parser.add_argument("--outfile", required=True, help="Output CSV file")
    parser.add_argument(
        "--reader-type",
        choices=["ribotish", "ribowaltz"],
        default="ribotish",
        help="Reader type for profile file (default: ribotish)",
    )
    parser.add_argument(
        "--consequence",
        default="frameshift",
        help="Variant consequence to filter (default: frameshift)",
    )
    parser.add_argument(
        "--no-chi2",
        action="store_true",
        help="Skip chi-square test computation",
    )
    args = parser.parse_args()

    vc = VariantCollector(args.vcf)
    tc = TranscriptCollector(args.gtf)

    if args.reader_type == "ribotish":
        reader = RibotishReader(args.profile)
    else:
        reader = RiboWaltzReader(args.profile)

    builder = DatasetBuilder(
        variant_collector=vc,
        transcript_collector=tc,
        ribotish_reader=reader,
        consequence=args.consequence,
    )

    df = builder.to_pandas(with_chi2=not args.no_chi2)
    df.to_csv(args.outfile, index=False)
    print(f"Exported {len(df)} variants to {args.outfile}")


if __name__ == '__main__':
    main()
