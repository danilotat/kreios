"""
This module defines the VariantExtended class for encapsulating and processing
cyvcf2.Variant objects.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Union
import numpy as np
import pysam
from cyvcf2 import Variant
from metrics import CoverageMetrics

class VariantExtended:
    """
    An extended variant class to encapsulate a cyvcf2.Variant object
    and provide methods for coverage metrics calculation.
    """

    def __init__(self, variant: Variant):
        if not isinstance(variant, Variant):
            raise TypeError("variant must be an instance of cyvcf2.Variant")
        self.variant = variant
        self.chrom = variant.CHROM
        self.pos = variant.POS  # 1-based
        self.ref = variant.REF
        self.alts = variant.ALT
        self.is_het = self._read_genotype()[0]
        self.is_homalt = self._read_genotype()[1]
        self.is_homref = self._read_genotype()[2]
        self.is_haploid = self._read_genotype()[3]
        self.is_phased = self._read_genotype()[4]

    def _read_genotype(self):
        """
        Just an helper to handle haploid scenario. Motivated by
        https://github.com/brentp/cyvcf2/issues/204
        """
        # unpack genotype
        is_het = False
        is_homalt = False
        is_homref = False
        is_haploid = False
        phased = False
        fields = self.variant.genotypes[0]
        if len(fields) == 2:
            # haploid call
            is_haploid = True
            is_homalt = True if fields[0] == 1 else False
            is_homref = True if fields[0] == 0 else False
        else:
            # diploid call
            is_homalt = True if fields[0] == 1 and fields[1] == 1 else False
            is_het = True if fields[0] != fields[1] else False
            is_homref = True if fields[0] == 0 and fields[1] == 0 else False
        return (is_het, is_homalt, is_homref, is_haploid, phased)

    @property
    def is_snp(self) -> bool:
        """Checks if the variant is a single nucleotide polymorphism."""
        return True if self.variant.is_snp else False

    @property
    def is_insertion(self) -> bool:
        """Checks if the variant is an insertion."""
        return True if len(self.ref) < len(self.alts[0]) else False

    @property
    def is_deletion(self) -> bool:
        """Checks if the variant is a deletion."""
        return True if len(self.ref) > len(self.alts[0]) else False

    def compute_coverage_metrics(
        self, bam_file: pysam.AlignmentFile
    ) -> Union[CoverageMetrics, None]:
        """Public method to compute coverage metrics based on the variant type."""
        if self.is_snp:
            return self._compute_snp_metrics(bam_file)
        elif self.is_insertion:
            return self._compute_insertion_metrics(bam_file)
        elif self.is_deletion:
            return self._compute_deletion_metrics(bam_file)
        else:
            logging.warning(
                f"MNV are still not supported. Skipping variant {self.chrom}:{self.pos} {self.ref}>{self.alts}"
            )
            return None

    def _compute_snp_metrics(self, bam_file: pysam.AlignmentFile) -> CoverageMetrics:
        """Computes coverage metrics for a single nucleotide variant."""
        pos = self.pos - 1  # 0-based
        covs = bam_file.count_coverage(
            self.chrom, int(pos), int(pos) + 1, read_callback="all"
        )
        # order is A C G T
        covs = {
            k: v
            for k, v in zip("ACGT", [covs[0][0], covs[1][0], covs[2][0], covs[3][0]])
        }
        ac = {base: covs.get(base, 0) for base in [self.ref] + self.alts}
        dp = sum(covs.values())
        all_alleles = [self.ref] + self.alts
        all_bqs = defaultdict(list)
        all_mqs = defaultdict(list)
        # fetch for base qualities and mapping qualities
        for pileupcolumn in bam_file.pileup(
            self.chrom,
            int(pos),
            int(pos) + 1,
            truncate=True,
            max_depth=0,
            stepper="all",
        ):
            if pileupcolumn.pos != pos:
                continue
            for pileupread in pileupcolumn.pileups:
                if pileupread.is_del or pileupread.is_refskip:
                    # here we're just working with mapped bases
                    continue
                base = pileupread.alignment.query_sequence[pileupread.query_position]
                if base not in all_alleles:
                    continue
                all_bqs[base].append(
                    pileupread.alignment.query_qualities[pileupread.query_position]
                )
                all_mqs[base].append(pileupread.alignment.mapping_quality)
        median_bqs = {
            base: int(np.median(all_bqs[base])) if all_bqs[base] else None
            for base in all_alleles
        }
        median_mqs = {
            base: int(np.median(all_mqs[base])) if all_mqs[base] else None
            for base in all_alleles
        }
        return CoverageMetrics(
            ac=ac,
            dp=dp,
            bqs=median_bqs,
            mqs=median_mqs,
            all_bqs=dict(all_bqs),
            all_mqs=dict(all_mqs),
        )

    def _compute_insertion_metrics(
        self, bam_file: pysam.AlignmentFile
    ) -> CoverageMetrics:
        """Computes coverage metrics for an insertion by parsing the CIGAR string."""
        pileup_pos = self.pos - 1  # Convert to 0-based
        insertion_alt = self.alts[0]
        # Extract just the inserted sequence (without anchor base)
        insertion_seq = insertion_alt[len(self.ref) :]
        insertion_len = len(insertion_seq)
        all_alleles = [self.ref, insertion_alt]
        all_mqs = defaultdict(list)
        all_bqs = defaultdict(list)

        # Calculate coverage at anchor position for DP
        covs = bam_file.count_coverage(
            self.chrom, int(pileup_pos), int(pileup_pos) + 1, read_callback="all"
        )
        # Order is A C G T
        covs = {
            k: v
            for k, v in zip("ACGT", [covs[0][0], covs[1][0], covs[2][0], covs[3][0]])
        }
        dp = sum(covs.values())
        # Iterate through reads at the anchor position
        for pileupcolumn in bam_file.pileup(
            self.chrom, pileup_pos, pileup_pos + 1, truncate=True, max_depth=0
        ):
            if pileupcolumn.pos != pileup_pos:
                continue

            for pileupread in pileupcolumn.pileups:
                read = pileupread.alignment
                found_insertion = False

                # Only check reads that have an indel
                if pileupread.indel > 0:  # Positive indel means insertion
                    current_ref_pos = read.reference_start  # 0-based
                    current_query_pos = 0

                    # Parse CIGAR string to find insertions
                    for op, length in read.cigartuples:
                        # Check for insertion at current reference position
                        if op == 1:  # Insertion
                            # Insertion occurs AFTER current_ref_pos
                            # For VCF position (1-based), insertion is after pos-1 (0-based)
                            # So we check if current_ref_pos == pileup_pos + 1
                            if (
                                current_ref_pos == pileup_pos + 1
                                and length == insertion_len
                            ):
                                # Extract the inserted bases from the read
                                inserted_bases = read.query_sequence[
                                    current_query_pos : current_query_pos + length
                                ]
                                # Check if it matches our expected insertion
                                if inserted_bases == insertion_seq:
                                    all_mqs[insertion_alt].append(read.mapping_quality)
                                    # Extract base qualities for inserted bases
                                    inserted_quals = read.query_qualities[
                                        current_query_pos : current_query_pos + length
                                    ]
                                    all_bqs[insertion_alt].extend(inserted_quals)
                                    found_insertion = True

                        # Update reference position for operations that consume reference
                        if op in {0, 2, 3, 7, 8}:  # M, D, N, =, X
                            current_ref_pos += length

                        # Update query position for operations that consume query
                        if op in {0, 1, 4, 7, 8}:  # M, I, S, =, X
                            current_query_pos += length

                        # Stop if we found the insertion or passed the position
                        if found_insertion or current_ref_pos > pileup_pos + 1:
                            break

                # If no insertion found, this read supports the reference
                if not found_insertion:
                    all_mqs[self.ref].append(read.mapping_quality)
                    # Get base quality at the reference position
                    if not pileupread.is_del and not pileupread.is_refskip:
                        all_bqs[self.ref].append(
                            read.query_qualities[pileupread.query_position]
                        )

        # Calculate allele counts from mapping quality lists
        ac = {base: len(all_mqs[base]) for base in all_alleles}

        # Calculate median mapping qualities
        median_mqs = {
            base: int(np.median(all_mqs[base])) if all_mqs[base] else None
            for base in all_alleles
        }

        # Calculate median base qualities
        median_bqs = {
            base: int(np.median(all_bqs[base])) if all_bqs[base] else None
            for base in all_alleles
        }

        return CoverageMetrics(
            ac=ac,
            dp=dp,
            bqs=median_bqs,
            mqs=median_mqs,
            all_bqs=dict(all_bqs),
            all_mqs=dict(all_mqs),
        )

    def _compute_deletion_metrics(
        self, bam_file: pysam.AlignmentFile
    ) -> CoverageMetrics:
        """Computes coverage metrics for a deletion by parsing the CIGAR string."""
        pileup_pos = self.pos - 1
        deletion_alt = self.alts[0]
        deletion_len = len(self.ref) - len(deletion_alt)
        all_alleles = [self.ref, deletion_alt]
        all_mqs = defaultdict(list)
        all_bqs = defaultdict(list)  # Add base qualities tracking
        dp = 0

        for pileupcolumn in bam_file.pileup(
            self.chrom, pileup_pos, pileup_pos + 1, truncate=True, max_depth=0
        ):
            if pileupcolumn.pos != pileup_pos:
                continue
            dp = pileupcolumn.n

            for pileupread in pileupcolumn.pileups:
                read = pileupread.alignment

                if pileupread.indel != 0:
                    current_ref_pos = read.reference_start  # 0-based
                    current_query_pos = 0
                    found_deletion = False
                    for op, length in read.cigartuples:
                        # Operation is a Deletion (D)
                        if op == 2:
                            # The deletion CIGAR starts at the variant position
                            if current_ref_pos == self.pos and length == deletion_len:
                                all_mqs[deletion_alt].append(read.mapping_quality)
                                # For deletions, get quality of flanking bases
                                # Get quality of base before deletion (anchor base)
                                if current_query_pos > 0:
                                    all_bqs[deletion_alt].append(
                                        read.query_qualities[current_query_pos - 1]
                                    )
                                found_deletion = True
                                break  # Found our variant in this read
                        if op in {0, 2, 3, 7, 8}:  # M, D, N, =, X (consumes reference)
                            current_ref_pos += length
                        if op in {0, 1, 4, 7, 8}:  # M, I, S, =, X (consumes query)
                            current_query_pos += length
                        if current_ref_pos > self.pos:
                            break
                    if not found_deletion:
                        all_mqs[self.ref].append(read.mapping_quality)
                        # Get base quality at the reference position
                        if not pileupread.is_del and not pileupread.is_refskip:
                            all_bqs[self.ref].append(
                                read.query_qualities[pileupread.query_position]
                            )
                else:  # No indel, this read supports the reference
                    all_mqs[self.ref].append(read.mapping_quality)
                    # Get base quality at the reference position
                    if not pileupread.is_del and not pileupread.is_refskip:
                        all_bqs[self.ref].append(
                            read.query_qualities[pileupread.query_position]
                        )

        ac = {base: len(all_mqs[base]) for base in all_alleles}
        median_mqs = {
            base: int(np.median(all_mqs[base])) if all_mqs[base] else None
            for base in all_alleles
        }
        median_bqs = {
            base: int(np.median(all_bqs[base])) if all_bqs[base] else None
            for base in all_alleles
        }

        return CoverageMetrics(
            ac=ac,
            dp=dp,
            bqs=median_bqs,
            mqs=median_mqs,
            all_bqs=dict(all_bqs),
            all_mqs=dict(all_mqs),
        )
