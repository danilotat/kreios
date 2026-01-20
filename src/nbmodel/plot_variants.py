#!/usr/bin/env python3
"""
Plot frameshift signal for variants listed in a CSV file.

Usage:
    python plot_variants.py --gtf annotation.gtf --csv variants.csv --profile ribotish.transprofile --outdir plots/
"""
import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from gtf import TranscriptCollector
from reader import RibotishReader
from plotting import plot_frameshift_signal


def sanitize_filename(variant_id: str) -> str:
    """Convert variant_id to a safe filename."""
    # Replace characters that are problematic in filenames
    return re.sub(r'[:<>|"?*]', '_', variant_id)


def main():
    parser = argparse.ArgumentParser(
        description="Plot frameshift signal for variants from a CSV file"
    )
    parser.add_argument(
        "--gtf", required=True,
        help="GTF annotation file (for TranscriptCollector)"
    )
    parser.add_argument(
        "--csv", required=True,
        help="CSV file with variants (must have tid, variant_id, variant_rel_pos columns)"
    )
    parser.add_argument(
        "--profile", required=True,
        help="Ribotish transprofile file"
    )
    parser.add_argument(
        "--outdir", default=".",
        help="Output directory for plots (default: current directory)"
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)

    # Load data sources
    print(f"Loading GTF from {args.gtf}...")
    tc = TranscriptCollector(args.gtf)

    print(f"Loading ribotish profile from {args.profile}...")
    reader = RibotishReader(args.profile)

    print(f"Reading variants from {args.csv}...")
    df = pd.read_csv(args.csv)
    
    # infer sname 
    sname = os.path.basename(args.csv).split('.')[0]
    # Validate required columns
    required_cols = ['tid', 'variant_id', 'variant_rel_pos']
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    print(f"Found {len(df)} variants to plot")

    # Plot each variant
    for idx, row in df.iterrows():
        tid = row['tid']
        variant_id = row['variant_id']
        variant_rel_pos = int(row['variant_rel_pos'])

        # Get transcript boundaries
        tid_start, tid_end, strand = tc[tid]
        if tid_start is None or tid_end is None:
            print(f"  Skipping {variant_id}: transcript {tid} not found in GTF")
            continue

        transcript_len = tid_end - tid_start

        # Extract profiles from ribotish
        _, pre_profile = reader._get_profile(tid, 0, variant_rel_pos)
        _, after_profile = reader._get_profile(tid, variant_rel_pos, transcript_len)

        if len(pre_profile) == 0 and len(after_profile) == 0:
            print(f"  Skipping {variant_id}: no profile data for {tid}")
            continue

        # Get features for annotation
        features = tc.get_relative_features(tid)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 4))
        axs = plot_frameshift_signal(
            pre_profile=pre_profile,
            after_profile=after_profile,
            variant_rel_pos=variant_rel_pos,
            features=features,
            ax=ax,
            title=f"{tid} - {variant_id}",
            show_frames=True,
        )
        axs[0].set_xlim(max(variant_rel_pos-200,0), variant_rel_pos+200)
        axs[1].set_xlim(max(variant_rel_pos-200,0), variant_rel_pos+200)
        # Save plot
        safe_name = sanitize_filename(variant_id)
        outpath = os.path.join(args.outdir, f"{sname}_{safe_name}.pdf")
        fig.savefig(outpath, bbox_inches='tight', dpi=350, transparent=True)
        plt.close(fig)

        print(f"  [{idx + 1}/{len(df)}] Saved {outpath}")

    print("Done!")


if __name__ == '__main__':
    main()
