"""
This module contains the VCF header definitions for the ribo-seq annotation.
"""

HEADERS = [
    {
        "ID": "ribo_af",
        "Description": f"Allele frequency for the alternate alleles in the ribo sample",
        "Type": "Float",
        "Number": "A",
    },
    {
        "ID": "ribo_dp",
        "Description": "Total depth of coverage in the ribo sample/s (independent of alleles)",
        "Type": "Float",
        "Number": "1",
    },
    {
        "ID": "ribo_ac",
        "Description": "Allele count for the alternate alleles in the ribo sample/s",
        "Type": "Integer",
        "Number": "A",
    },
    {
        "ID": "ribo_pu",
        "Description": "Probability of detecting a mutation given the observed supporting reads (AC), "
        "the observed total coverage (DP) and the expected VAF in the ribo sample/s",
        "Type": "Float",
        "Number": "A",
    },
    {
        "ID": "ribo_pw",
        "Description": "Power to detect a somatic mutation as described in Absolute "
        "given the observed total coverage (DP) "
        "and the provided tumor purity and ploidies in the ribo sample/s",
        "Type": "Float",
        "Number": "1",
    },
    {
        "ID": "ribo_k",
        "Description": "Minimum number of supporting reads, k, such that the probability of observing "
        "k or more non-reference reads due to sequencing error is less than the defined FPR "
        "in the ribo sample/s",
        "Type": "Float",
        "Number": "1",
    },
    {
        "ID": "ribo_bq",
        "Description": "Median base call quality of the reads supporting each allele in the "
        "ribo sample/s",
        "Type": "Float",
        "Number": "1",
    },
    {
        "ID": "ribo_mq",
        "Description": "Median mapping quality of the reads supporting each allele in the "
        "ribo sample/s",
        "Type": "Float",
        "Number": "1",
    },
    {
        "ID": "ribo_rsmq",
        "Description": "Rank sum test comparing the MQ distributions supporting the reference and the "
        "alternate in the ribo sample/s. Identical distributions will have a value of 0, larger "
        "values away from 0 indicate different distributions.",
        "Type": "Float",
        "Number": "A",
    },
    {
        "ID": "ribo_rsmq_pv",
        "Description": "Rank sum test comparing the mapping quality distributions between alternate "
        "and reference p-value in the ribo sample/s. , The null hypothesis is that there is no "
        "difference between the distributions",
        "Type": "Float",
        "Number": "A",
    },
    {
        "ID": "ribo_rsbq",
        "Description": "Rank sum test comparing the base call qualities distributions supporting the reference "
        "and the alternate in the ribo sample/s. Identical distributions will have a value of 0, "
        "larger values away from 0 indicate different distributions.",
        "Type": "Float",
        "Number": "A",
    },
    {
        "ID": "ribo_rsbq_pv",
        "Description": "Rank sum test comparing the base call qualities distributions between alternate "
        "and reference p-value in the ribo sample/s. , The null hypothesis is that there is no "
        "difference between the distributions",
        "Type": "Float",
        "Number": "A",
    },
    {
        "ID": "kmer_seq",
        "Description": "k-mer used for inferring the number of reads supporting the variant in the riboseq sample",
        "Type": "String",
        "Number": "A",
    },
    {
        "ID": "kmer_entropy",
        "Description": "Shannon entropy of the kmer in use",
        "Type": "Float",
        "Number": "A",
    },
    {
        "ID": "kmer_ribo_reads",
        "Description": "Number of reads supporting the kmer used for Indel",
        "Type": "Float",
        "Number": "A",
    },
]
