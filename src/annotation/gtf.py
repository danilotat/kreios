from typing import Tuple, List, Dict, Union
from intervaltree import IntervalTree

class GTF_record(object):
    """
    Optimized GTF record parser.
    """
    __slots__ = ('chromosome', 'source', 'feature_type', 'start', 'end', 
                 'score', 'strand', 'phase', 'length', 'attributes')

    def __init__(
        self,
        chromosome,
        source,
        feature_type,
        start,
        end,
        score,
        strand,
        phase,
        attributes,
    ):
        self.chromosome = str(chromosome)
        self.source = source
        self.feature_type = feature_type
        # GTF is 1-based, converting to 0-based for BED/pysam
        self.start = int(start) - 1
        self.end = int(end)
        self.score = score
        self.strand = strand
        self.phase = phase
        self.length = abs(self.end - self.start)
        self.attributes = self._parse_attributes(attributes)

    def _parse_attributes(self, attributes: Union[str, dict]) -> dict:
        if isinstance(attributes, dict):
            return attributes
        
        attr_dict = {}
        parts = attributes.strip().split(';')
        for part in parts:
            part = part.strip()
            if not part:
                continue
            try:
                key, value = part.split(' ', 1)
                attr_dict[key] = value.strip('"')
            except ValueError:
                continue
        return attr_dict

    @property
    def is_coding(self) -> bool:
        return self.attributes.get("gene_type", "") == "protein_coding"

class TranscriptBlock(object):
    """
    Dict-like BED12 block object. Fields are accessible as attributes and via
    mapping operations (obj['chrom'] = 'chr1', etc.).
    Note: blockCount, blockSizes and blockStarts are lists by design.
    FIELDS = [
        'chrom', 'start', 'end', 'name', 'score', 'strand',
        'thickStart', 'thickEnd', 'itemRgb',
        'blockCount', 'blockSizes', 'blockStarts'
    ]
    """
    __TRANSCRIPT_FIELDS__ = ['chrom', 'start', 'end', 'name', 'score', 'strand']
    __EXPANDED_FIELDS__ = [
        'chrom', 'start', 'end', 'name', 'score', 'strand',
        'thickStart', 'thickEnd', 'itemRgb',
        'blockCount', 'blockSizes', 'blockStarts'
    ]
    def __init__(self, **kwargs):
        for f in self.__TRANSCRIPT_FIELDS__:
            v = kwargs.get(f, None)
            object.__setattr__(self, f, v)
        self.blocks = IntervalTree()
        
    def __getitem__(self, key):
        if key not in self.__EXPANDED_FIELDS__ + ['blocks']:
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key, value):
        if key == 'blocks':
            if not isinstance(value, tuple):
                raise ValueError(f"Intervals could be added just by passing a tuple of (start, end)")
            else:
                self.blocks.addi(*value)  # Fixed: was self['blocks']
        elif key not in self.__EXPANDED_FIELDS__: 
                raise KeyError(key)
        else:
            setattr(self, key, value)
    
    def slice_range(self, pos_from: int, pos_to: int) -> Tuple[List, List]:
        """
        Slice according to the interval in question.
        The returned value is 
        starts, ends  
        """
        overlapping_interval = self.blocks.overlap(pos_from, pos_to)
        if not overlapping_interval:
            raise ValueError(f"No overlapping entries found in the provided range {pos_from}:{pos_to}")
        else:
            overlapping_interval = sorted(overlapping_interval)
            starts, ends = [s[0] for s in overlapping_interval], [s[1] for s in overlapping_interval]
            return starts, ends 

    def asBED12(self, pos_from: int, pos_to: int) -> str:
        bedfields = []
        starts, ends = self.slice_range(pos_from, pos_to)
        rel_starts = [s - self.get('start') for s in starts]
        for f in self.__EXPANDED_FIELDS__:
            if f in self.__TRANSCRIPT_FIELDS__:
                # these are conventional field 
                bedfields.append(str(self.get(f)))
            else:
                if f in ['thickStart', 'thickEnd']:
                    #NOTE: this is because we're using bedtools after, thus we don't really care about these fields. 
                    # https://bedtools.readthedocs.io/en/latest/content/general-usage.html#bed-format
                    bedfields.append(str(self.get('start')))
                elif f == 'itemRgb':
                    bedfields.append(str(0))
                elif f == 'blockCount':
                    bedfields.append(str(len(starts)))  # Fixed: convert to string
                elif f == 'blockSizes':
                    sizes = [str(e-s) for s, e in zip(starts, ends)]  # Fixed: was s-e (wrong order)
                    bedfields.append(",".join(sizes))
                elif f == 'blockStarts':
                    bedfields.append(",".join([str(s) for s in rel_starts]))  # Fixed: was .append() instead of .join()
        return "\t".join(bedfields)

    def get(self, key, default=None):
        return self[key] if key in self.__TRANSCRIPT_FIELDS__ else default

    def keys(self):
        return list(self.__TRANSCRIPT_FIELDS__)

    def items(self):
        return [(k, getattr(self, k)) for k in self.__TRANSCRIPT_FIELDS__]

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__TRANSCRIPT_FIELDS__}

    def __repr__(self):
        fields = ", ".join(f"{k}={repr(getattr(self,k))}" for k in self.__TRANSCRIPT_FIELDS__)
        return f"BED12_block({fields})"
 

class TranscriptCollector(object):
    """
    This class holds a TranscriptCollector which plays with the GTF to store
    a block representation of each transcript in order to play nicely with intervals
    thanks to intervaltree and to natively export in BED12 format
    """
    def __init__(self, gtf: str):
        self._gtf = gtf
        self.records = self._populate_records()
    
    def _populate_records(self):
        records = {}
        with open(self._gtf, 'r') as gtf:
            for line in gtf:
                if line.startswith("#"):
                    continue
                # Split line only once
                parts = line.rstrip().split('\t')
                if parts[2] == 'transcript':
                    entry = GTF_record(*parts)
                    tid = entry.attributes.get('transcript_id')
                    if not tid: 
                        continue
                    if tid not in records:
                        # instantiate the structure block
                        records[tid] = TranscriptBlock(
                            chrom=entry.chromosome,
                            start=entry.start,
                            end=entry.end,
                            strand=entry.strand,
                            name=tid,
                            score=entry.score,
                        )
                elif parts[2] in ['CDS', 'stop_codon']:
                    records[tid]['blocks'] = entry.start, entry.end
            return records
        
    def compose_BED12_string(self, tid: str, pos_from: int, pos_to: int) -> str:
        try:
            bedblock = self.records[tid]
            bed12_str = bedblock.asBED12(
                pos_from, pos_to
            )
            return bed12_str
        except KeyError:
            print(f"The transcript id {tid} is not present")