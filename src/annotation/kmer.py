import subprocess
import cyvcf2
import os
import pysam
import numpy as np
from collections import Counter
from scipy.stats import entropy
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)


class IndelVariant:
    def __init__(
        self,
        variant: cyvcf2.Variant,
        vcf: str,
        genome: str,
        outfolder: str,
        flanking=50,
        kmersize=17,
    ):
        self.variant = variant
        self.vcf = vcf
        self.flanking = flanking
        self.region = f"{variant.CHROM}:{int(variant.POS) - flanking}-{int(variant.POS) + flanking}"
        self.genome = genome
        self.outname = self.region.replace(":", "_").replace("-", "_")
        self.outfile = os.path.join(outfolder, f"{self.outname}.fasta")
        self.kmersize = kmersize
        self.kmer = self.generate_kmer()

    @property
    def entropy(self) -> float:
        """Calculate Shannon entropy of the kmer"""
        length = len(self.kmer)
        if length == 0:
            return 0.0
        counts = Counter(self.kmer.upper()).values()
        return np.round(entropy(list(counts), base=2), 4)

    def generate_kmer(self) -> str:
        """
        Generate kmer centered on the variant position, ensuring it includes at least one alt base
        """
        # Get reference sequence
        ref_seq = (
            pysam.FastaFile(self.genome)
            .fetch(
                self.variant.CHROM,
                int(self.variant.POS) - self.flanking - 1,
                int(self.variant.POS) + self.flanking,
            )
            .upper()
        )

        # Generate consensus with variant
        cmd = f"samtools faidx {self.genome} {self.region} | bcftools consensus -s - {self.vcf} -o {self.outfile}"
        subprocess.run(cmd, shell=True, check=True)
        alt_seq = []
        with open(self.outfile, "r") as fasta:
            for line in fasta:
                if not line.startswith(">"):
                    alt_seq.append(line.rstrip())
        alt_seq = "".join(alt_seq).upper()
        # drop the intermediate file
        os.remove(self.outfile)
        # The variant is at position self.flanking in the reference
        ref_var_pos = self.flanking
        # For the alt sequence, we need to account for indel length difference
        ref_len = len(self.variant.REF)
        alt_len = len(self.variant.ALT[0])
        size_diff = alt_len - ref_len
        # Alt variant position is the same as ref, but sequence length differs
        alt_var_pos = self.flanking
        # Extract kmer that includes at least one base from the variant
        # Center the kmer on the first alt base (right after the shared reference base)
        kmer_start = alt_var_pos - (self.kmersize // 2)
        kmer_end = kmer_start + self.kmersize
        # Ensure we don't go out of bounds
        if kmer_start < 0:
            kmer_start = 0
            kmer_end = self.kmersize
        elif kmer_end > len(alt_seq):
            kmer_end = len(alt_seq)
            kmer_start = max(0, kmer_end - self.kmersize)
        kmer = alt_seq[kmer_start:kmer_end]
        # Validate kmer length
        if len(kmer) < self.kmersize:
            raise ValueError(
                f"Kmer too short ({len(kmer)} bp): flanking region may be too small for this variant"
            )
        return kmer

    def count_kmer(self, jellyfish_db: str):
        count = subprocess.run(
            ["jellyfish", "query", jellyfish_db, self.kmer],
            capture_output=True,
            text=True,
        )
        return int(count.stdout.rstrip().split(" ")[-1])

    def __repr__(self):
        return f"Indel on {self.region}:{self.variant.REF}>{self.variant.ALT[0]} searching for kmer {self.kmer}"


if __name__ == "__main__":
    vcf = "/leonardo_scratch/fast/IscrC_B-CARE/RNA/Chotani_2022/vcf/phased_vep/annotated/SRR15513261_GSM5527786.deepvariant.phased.vep.riboannotated.vcf.gz"
    jellyfish_db = "/leonardo_scratch/fast/IscrC_B-CARE/ribo/Chotani_2022/kmers/SRR15513165_GSM5527690_Fibroblast_21_kmer17.jf"
    genome = "/leonardo_work/IscrC_B-CARE/resources/GRCh38/GRCh38_GIABv3_no_alt_analysis_set_maskedGRC_decoys_MAP2K3_KMT2C_KCNJ18.fasta"
    for variant in cyvcf2.VCF(vcf):
        if not variant.is_snp:
            if int(variant.INFO.get("ribo_dp")) > 10:
                indel = IndelVariant(
                    variant=variant,
                    vcf=vcf,
                    genome=genome,
                    outfolder="/leonardo_scratch/fast/IscrC_B-CARE/RNA/Chotani_2022/vcf/phased_vep/annotated/kmers",
                )
                if indel.entropy < 1:
                    logging.warning(
                        f"The indel has very low entropy {indel.entropy}. Results may be wrong."
                    )
                count = indel.count_kmer(jellyfish_db)
                if count > 1 and indel.entropy > 1:
                    logging.info(
                        f"Count: {count} of {indel.kmer} with entropy {indel.entropy} in {variant.INFO.get("ribo_dp")} given AF={variant.gt_alt_freqs[0]}"
                    )
