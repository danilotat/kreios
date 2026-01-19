from typing import Union
import logging

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
        # GTF is 1-based, converting to 0-based 
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
    
class TranscriptCollector():
    TRACKED_FEATURES = {'start_codon', 'CDS'}

    def __init__(self, gtf: str):
        self._gtf = gtf
        self.transcripts = self.read_gtf()

    def read_gtf(self):
        transcripts = {}
        with open(self._gtf, 'r') as gtf:
            for line in gtf:
                if line.startswith('#'):
                    continue
                entry = GTF_record(*line.rstrip().split('\t'))
                tid = entry.attributes.get('transcript_id', None)
                if not tid:
                    continue
                tid = tid.split('.')[0]

                if entry.feature_type == 'transcript':
                    if tid not in transcripts:
                        transcripts[tid] = {
                            'start': entry.start,
                            'end': entry.end,
                            'strand': entry.strand,
                            'features': []
                        }
                    else:
                        transcripts[tid]['start'] = entry.start
                        transcripts[tid]['end'] = entry.end
                        transcripts[tid]['strand'] = entry.strand

                elif entry.feature_type in self.TRACKED_FEATURES:
                    if tid not in transcripts:
                        transcripts[tid] = {
                            'start': None,
                            'end': None,
                            'strand': entry.strand,
                            'features': []
                        }
                    transcripts[tid]['features'].append(
                        (entry.feature_type, entry.start, entry.end)
                    )
        return transcripts

    def get_relative_features(self, tid):
        """
        Get features with coordinates relative to transcript 5' end.
        Returns list of (feature_type, rel_start, rel_end) tuples.

        For forward strand: relative to genomic start (lower coordinate)
        For reverse strand: relative to genomic end (higher coordinate, which is the 5' end)
        """
        if tid not in self.transcripts:
            logging.warning(f"{tid} not present in the GTF")
            return []

        tx = self.transcripts[tid]
        tx_start = tx['start']
        tx_end = tx['end']
        strand = tx['strand']
        if tx_start is None:
            return []

        relative_features = []
        for feat_type, start, end in tx['features']:
            if strand == '+':
                rel_start = start - tx_start
                rel_end = end - tx_start
            else:  # reverse strand
                # For reverse strand, measure from the 5' end (tx_end)
                rel_start = tx_end - end
                rel_end = tx_end - start
            relative_features.append((feat_type, rel_start, rel_end))

        return relative_features

    def __getitem__(self, tid):
        try:
            tx = self.transcripts[tid]
            return (tx['start'], tx['end'], tx['strand'])
        except KeyError:
            logging.warning(f"{tid} not present in the GTF")
            return (None, None, None)

    def get_start_codon_pos(self, tid):
        """
        Get the start codon position for a transcript.
        Returns the start position of the start_codon feature, or None if not found.
        """
        if tid not in self.transcripts:
            return None
        features = self.transcripts[tid].get('features', [])
        for feat_type, start, end in features:
            if feat_type == 'start_codon':
                return start
        return None


if __name__ == '__main__':
    tc = TranscriptCollector("/Users/danilo/Research/Tools/kreios/examples/gencode.v48.annotation.gtf")
    print("Transcript coords:", tc['ENST00000832829'])
    print("Features (absolute):", tc.transcripts.get('ENST00000832829', {}).get('features', []))
    print("Features (relative):", tc.get_relative_features('ENST00000832829'))
    print("Missing transcript:", tc['TUAMAMMA'])