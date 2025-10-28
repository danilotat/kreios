"""
Main script to annotate a VCF file with ribosome profiling coverage metrics.
"""

import argparse
import logging
import os
import sys
import numpy as np
import pysam
from cyvcf2 import VCF, Writer
from kmer import IndelVariant
from metrics import PowerCalculator
from utils import build_jellyfish_index, calculate_rank_sum_test
from headers import HEADERS
from variant import VariantExtended


def setup_logging():
    """Sets up the logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def process_variant(
    variant: VCF,
    var_ext: VariantExtended,
    bam: pysam.AlignmentFile,
    jellyfish_db: str,
    args: argparse.Namespace,
    out_folder: str,
):
    """
    Processes a single variant, calculates metrics, annotates it, and returns it.
    Returns None if the variant cannot be processed.
    """
    ribo_stats = var_ext.compute_coverage_metrics(bam)
    if not ribo_stats:
        logging.warning(
            f"Could not compute metrics for variant {variant.CHROM}:{variant.POS}; skipping."
        )
        return None

    # Instantiate the power calculator for this variant's genotype
    power_calc = PowerCalculator(expected_vaf=0.5 if var_ext.is_het else 1.0)

    # --- Calculate Core Metrics ---
    try:
        af = np.round((ribo_stats.ac.get(variant.ALT[0], 0) / ribo_stats.dp), 3)
    except ZeroDivisionError:
        af = 0.0

    ribo_pu = power_calc.calculate_power(
        dp=ribo_stats.dp, ac=ribo_stats.ac.get(variant.ALT[0], 0)
    )
    power, k = power_calc.calculate_absolute_power(dp=ribo_stats.dp)
    ribo_pw = power
    ribo_k = k

    # --- Perform Rank-Sum Tests ---
    alt_mqs = ribo_stats.all_mqs.get(variant.ALT[0], [])
    ref_mqs = ribo_stats.all_mqs.get(variant.REF, [])
    ribo_rsmq, ribo_rsmq_pv = calculate_rank_sum_test(alt_mqs, ref_mqs)

    alt_bqs = ribo_stats.all_bqs.get(variant.ALT[0], [])
    ref_bqs = ribo_stats.all_bqs.get(variant.REF, [])
    ribo_rsbq, ribo_rsbq_pv = calculate_rank_sum_test(alt_bqs, ref_bqs)

    # --- Handle INDELs with k-mers ---
    kmer_entropy, kmer_seq, kmer_ribo_reads = '.', '.', '.'
    if not var_ext.is_snp:
        indel_var = IndelVariant(
            variant,
            vcf=args.vcf,
            genome=args.genome,
            outfolder=out_folder,
            flanking=args.flanking_size,
            kmersize=args.kmer_size,
        )
        kmer_entropy = indel_var.entropy
        kmer_seq = indel_var.kmer
        kmer_ribo_reads = indel_var.count_kmer(jellyfish_db)

    # --- Annotate the VCF Record ---
    variant.INFO["ribo_dp"] = ribo_stats.dp
    variant.INFO["ribo_af"] = af
    variant.INFO["ribo_ac"] = ribo_stats.ac.get(variant.ALT[0], 0)
    variant.INFO["ribo_pu"] = ribo_pu
    variant.INFO["ribo_pw"] = ribo_pw
    variant.INFO["ribo_k"] = ribo_k
    variant.INFO["ribo_bq"] = ribo_stats.bqs.get(variant.ALT[0]) or 0
    variant.INFO["ribo_mq"] = ribo_stats.mqs.get(variant.ALT[0]) or 0
    variant.INFO["ribo_rsmq"] = ribo_rsmq
    variant.INFO["ribo_rsmq_pv"] = ribo_rsmq_pv
    variant.INFO["ribo_rsbq"] = ribo_rsbq
    variant.INFO["ribo_rsbq_pv"] = ribo_rsbq_pv
    variant.INFO["kmer_seq"] = kmer_seq
    variant.INFO["kmer_entropy"] = kmer_entropy
    variant.INFO["kmer_ribo_reads"] = kmer_ribo_reads
    return variant


def main():
    """Main function to parse arguments and run the VCF annotation workflow."""
    parser = argparse.ArgumentParser(
        description="Annotate VCF with ribosome profiling coverage metrics."
    )
    parser.add_argument(
        "-v",
        "--vcf",
        type=str,
        required=True,
        help="Input VCF file (bgzipped and indexed).",
    )
    parser.add_argument(
        "-b",
        "--bam",
        type=str,
        required=True,
        help="Input BAM file with ribosome profiling data.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output annotated VCF file (bgzipped).",
    )
    parser.add_argument(
        "--fastq", required=True, help="Path to the riboseq fastq file."
    )
    parser.add_argument(
        "--genome",
        required=True,
        help="Path to the reference genome FASTA file (for INDELs).",
    )
    parser.add_argument(
        "-f",
        "--flanking_size",
        type=int,
        default=50,
        help="Size of the flanking region for INDELs.",
    )
    parser.add_argument(
        "-k", "--kmer_size", type=int, default=17, help="Size of the k-mer for INDELs."
    )
    args = parser.parse_args()

    setup_logging()

    # \\ Setup VCF and Jellyfish
    vcf_in = VCF(args.vcf)
    for header in HEADERS:
        vcf_in.add_info_to_header(header)

    out_folder = os.path.dirname(os.path.abspath(args.output))
    fname = vcf_in.samples[0]

    logging.info("Building or verifying Jellyfish k-mer index...")
    jellyfish_db = build_jellyfish_index(
        args.fastq,
        args.kmer_size,
        outfile=os.path.join(out_folder, f"{fname}_kmer_{args.kmer_size}nt.jf"),
    )
    logging.info("Jellyfish index is ready.")

    # \\ Process VCF Records
    bam = pysam.AlignmentFile(args.bam, mode="rb")
    out_vcf = Writer(args.output, tmpl=vcf_in, mode="wz")
    logging.info(f"Starting to process variants from {args.vcf}...")
    for header in HEADERS:
        vcf_in.add_info_to_header(header)
    for variant in vcf_in:
        if variant.FILTER is not None and variant.FILTER != "PASS":
            continue
        var_ext = VariantExtended(variant)
        annotated_variant = process_variant(
            variant, var_ext, bam, jellyfish_db, args, out_folder
        )
        if annotated_variant:
            out_vcf.write_record(annotated_variant)
    logging.info(f"Annotation complete. Output written to {args.output}")


if __name__ == "__main__":
    main()
