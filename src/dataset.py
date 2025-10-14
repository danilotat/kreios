import pysam
import cyvcf2
from collections import namedtuple
import logging
import sys
import torch
import os
import pandas as pd
from torch.utils.data import Dataset


class VariantTensorDataset(Dataset):
    """
    Pytorch dataset for loading tensors and the according metadata file.
    Attributes:
        dataset_dir (str): The root directory containing the dataset.
        metadata (pd.DataFrame): A DataFrame containing metadata about
        the dataset, including file paths and labels.
    Methods:
        __len__():
            Returns the total number of samples in the dataset.
        __getitem__(idx):
            Retrieves the tensor and label for the given index.
    """

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        self.metadata = pd.read_csv(metadata_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get the file path and label for the requested index
        tensor_filename = self.metadata.iloc[idx]['file_path']
        label = self.metadata.iloc[idx]['label']

        # Construct the full path to the tensor file
        tensor_path = os.path.join(self.dataset_dir, tensor_filename)

        # Load the single tensor from disk
        tensor = torch.load(tensor_path)

        # Convert label to a tensor if needed
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return tensor, label_tensor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ])

coverage = namedtuple('Coverage', ['A', 'C', 'G', 'T'])


class VariantExtended():
    def __init__(self, variant: cyvcf2.Variant, genome: str,  bam_file: str, flanking=128):
        self.variant = variant
        self.flanking = flanking
        self.full_region = f"{variant.CHROM}:{variant.POS - (self.flanking / 2)}-{self.variant.POS + (self.flanking / 2)}"
        self.identifier = f"{variant.CHROM}:{variant.POS}"
        self.full_id = f"{variant.CHROM}:{variant.POS}:{variant.REF}>{variant.ALT[0]}"
        self.phased = variant.format('PS')  # phased is an array
        if self.phased:
            self.phased = [f"{variant.CHROM}:{j}" for j in variant.format('PS')[
                0]]
        self.bam_file = pysam.AlignmentFile(bam_file, 'rb')
        self.genome = pysam.FastaFile(genome)
        self.reference_sequence = self._get_sequence()
        self.has_support = self._get_support()

    def _get_sequence(self):
        sequence = self.genome.fetch(
            self.variant.CHROM, self.variant.POS -
            (self.flanking / 2), self.variant.POS + (self.flanking / 2)
        )
        return sequence

    def _get_support(self):
        # fetch reads in position and check if it has support in the matched aligned bam.
        bam_coverage = self.bam_file.count_coverage(
            self.variant.CHROM, self.variant.POS, self.variant.POS + 1, read_callback='all'
        )
        # arrays are in order A,C,G,T
        coverage_nt = coverage(*[bam_coverage[i][0] for i in range(4)])
        if getattr(coverage_nt, self.variant.ALT[0]) > 0:
            return True
        else:
            return False


class VariantCollector():
    """
    @TODO: implement a smart way to iterate here using __next__ and __iter__
    """

    def __init__(self, vcf: str, bam: str, genome: str):
        self.vcf = cyvcf2.VCF(vcf)
        self.bam = bam
        self.genome = genome
        self.variants = self._collect_variants()

    def _collect_variants(self):
        collector = {}
        variants = 0
        supported = 0
        not_supported = 0
        for variant in self.vcf:
            variants += 1
            if variant.is_snp:  # support just for SNP now
                variantExt = VariantExtended(
                    variant, self.genome, self.bam
                )
                if variantExt.has_support:
                    supported += 1
                else:
                    not_supported += 1
                # NOTE: this breaks the compatibility with the Tensor Dataset!!
                collector[variantExt.identifier] = {
                    'variant': variantExt,
                    'ID': variantExt.full_id,
                    'label': 1 if variantExt.has_support else 0
                }
        logging.info(f"Processed {variants} variants")
        logging.info(f"Found support for {supported}/{variants}")
        logging.info(
            f"The remaining unsupported are {not_supported}/{variants}")
        return collector

    def retrieve_edited_seq(self, variant: VariantExtended, flanking=200):
        identifier = f"{variant.variant.CHROM}:{variant.variant.POS}"
        obtained_var = self.variants[identifier]['variant']
        # check if we have some phased variants
        phased_vars = None
        if variant.phased:
            phased_vars = []
            for j in variant.phased:
                try:
                    phased_vars.append(self.variants[j]['variant'])
                except KeyError:
                    logging.warning(
                        f"Variant {j} could not be found. Maybe it was deleted due to some filtering?")

        # now work with replacement masks as tuples (pos: ALT)
        # Use 0-based start position from cyvcf2
        replacements = [(obtained_var.variant.start,
                         obtained_var.variant.ALT[0])]
        if phased_vars:
            for phased_var in phased_vars:
                replacements.append(
                    (phased_var.variant.start, phased_var.variant.ALT[0]))

        # Convert 1-based POS to 0-based for pysam fetch
        variant_start_0based = obtained_var.variant.start
        fetch_start = variant_start_0based - flanking
        fetch_end = variant_start_0based + flanking + \
            1  # +1 for the SNP position to be included
        reference_sequence = obtained_var.genome.fetch(
            reference=obtained_var.variant.CHROM,
            start=fetch_start,
            end=fetch_end
        )

        # Create a mutable list of characters for easier replacement
        ref_seq_list = list(reference_sequence)

        for pos_0based, alt_allele in replacements:
            relative_pos = pos_0based - fetch_start
            variant_key = f"{variant.variant.CHROM}:{pos_0based + 1}"
            expected_ref = self.variants[variant_key]['variant'].variant.REF

            if reference_sequence[relative_pos] != expected_ref:
                raise ValueError(
                    f"Reference mismatch at {variant_key}! "
                    f"Expected {expected_ref} but found {reference_sequence[relative_pos]} in the reference genome."
                )
            ref_seq_list[relative_pos] = alt_allele

        return "".join(ref_seq_list)

    def __repr__(self):
        tot_var = len(self.variants.keys())
        positives = len([j for j in self.variants.values() if j['label'] == 1])
        negatives = tot_var - positives
        repr_str = f"VariantCollector(N={tot_var}, {positives}/{tot_var} positives and {negatives}/{tot_var})"
        return repr_str


if __name__ == '__main__':
    import sys
    vcf = "/CTGlab/projects/ribo/Chotani_2022/RNAseq/VCF/SRR15513228.passOnly.vcf.gz"
    sname = os.path.basename(vcf).split('.')[0]
    bam = "/CTGlab/projects/ribo/Chotani_2022/ribo/SRR15513151_GSM5527676_Brain_4_RiboSeq_Homo_sapiens_RNA-Seq.genome.sorted.bam"
    genome = '/CTGlab/db/GRCh38_GIABv3_no_alt_analysis_set_maskedGRC_decoys_MAP2K3_KMT2C_KCNJ18.fasta'
    collected_vars = VariantCollector(vcf, bam, genome)
    print(collected_vars)
    ids, labels, seqs = [], [], []
    for variant_id in collected_vars.variants:
        extVar, ID,  label = collected_vars.variants[variant_id].values()
        retr_seq = collected_vars.retrieve_edited_seq(extVar)
        ids.append(f"{sname}_{ID}")
        labels.append(label)
        seqs.append(retr_seq)
    pd.DataFrame({
        'ids': ids,
        'seq': seqs,
        'label': labels
    }).to_csv(f'/CTGlab/projects/kreios/data/expl/{sname}_seqs_with_label_400nt.csv', index=False)
