import cyvcf2 
from collections import defaultdict
from dataclasses import dataclass
import logging 

@dataclass
class VariantObj:
    chrom: str
    pos: int
    ref: str
    alt: str
    tid: str
    gene: str
    is_snp: bool
    dp: int
    alt_dp: int
    filtered: bool
    consequence: str
    def __init__(self, variant: cyvcf2.Variant, csq: dict):
        self._variant = variant
        self.chrom = variant.CHROM
        self.pos = variant.POS - 1 #0-based for pysam compatibility 
        self.ref = variant.REF
        self.alt = variant.ALT[0]
        self.tid = csq.get('Feature', None)
        self.is_snp = True if variant.is_snp else False
        self.dp = variant.gt_depths[0]
        self.alt_dp = variant.gt_alt_depths[0]
        self.filtered = True if variant.FILTER is None else False
        self.consequence = csq.get('Consequence', None)
    
    @property
    def variant_id(self) -> str:
        """Compact variant representation: chr:pos:ref>alt"""
        return f"{self.chrom}:{self.pos}:{self.ref}>{self.alt}"

    def __repr__(self):
        return f"Variant=({self.variant_id}, {self.consequence}, {self.tid})"



class VariantCollector(object):
    def __init__(self, vcf: str):
        self._vcf = cyvcf2.VCF(vcf)
        self._csq_keys = [
            j.strip() for j in self._vcf.get_header_type('CSQ')['Description'].replace('"','').split('Format: ')[1].split('|')
        ]
        self._variants = self._read_vcf()
    

    def _read_vcf(self):
        variants = defaultdict(list)
        for variant in self._vcf:
            csq = variant.INFO.get('CSQ')
            if csq:
                csq = {k:v for k,v in zip(
                   self._csq_keys, csq.split('|') 
                )}
                tid = csq.get('Feature', None)
                if tid:
                    variants[tid].append(
                        VariantObj(variant, csq)
                    ) 
        return variants
    
    def get(self, tid, default=None):
        return self._variants.get(tid, default)

    def __getitem__(self, tid):
        if tid not in self._variants:
            logging.warning(f'{tid} is not present in the VariantCollector. Returning None.')
            return None
        return self._variants[tid]

    def __iter__(self):
        for variants in self._variants.values():
            yield from variants

    def __repr__(self):
        return f"VariantCollector with {len(self._variants)} transcripts."
    
if __name__ == '__main__':
    kek = VariantCollector("/Users/danilo/Research/Tools/kreios/examples/SRR20649716_GSM6395076.deepvariant.phased.vep.vcf.gz")
    print(kek)
    print(kek.get('ENST00000649529'))
    print(kek['ENST00000649529'])
    print(kek.get('FAKETRANSCRIPT'))


        