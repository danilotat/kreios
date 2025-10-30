from collections import defaultdict
from typing import Optional, Union, Iterable
from abc import ABC, abstractmethod
import logging

class GTF_record(object):
    """
    A GTF record is the first building block of the parser.
    The attribute field is parsed resulting in a dict.

    Attributes
    ----------
    chromosome : str
        The chromosome that the GTF record belongs to.
    source : str
        The source of the GTF record.
    feature_type : str
        The type of feature that the GTF record represents.
    start : int
        The start position of the feature in the chromosome.
    end : int
        The end position of the feature in the chromosome.
    score : str
        The score of the GTF record.
    strand : str
        The strand that the GTF record belongs to.
    phase : str
        The phase of the GTF record.
    length : int
        The length of the feature.
    attributes : dict
        A dictionary of attributes from the GTF record.

    Methods
    -------
    __init__(chromosome, source, feature_type, start, end, score, strand, phase, attributes)
        Initialize a GTF record object.
    parse_attributes(attributes)
        Parse the attributes of a GTF record.
    is_coding(feat_dict)
        Check if a GTF record is coding.
    """

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
        self.start = int(start)
        self.end = int(end)
        self.score = score
        self.strand = strand
        self.phase = phase
        self.length = abs(self.end - self.start)
        self.attributes = self._parse_attributes(attributes)

    def _parse_attributes(self, attributes) -> dict:
        if isinstance(attributes, dict):
            return attributes
        attr_dict = {}
        for item in attributes.strip().strip(";").split(";"):
            if not item.strip():
                continue
            # Each attribute is in the form: key "value"
            parts = item.strip().split(" ", 1)
            if len(parts) == 2:
                key, value = parts
                attr_dict[key] = value.strip('"')
        return attr_dict

    @property
    def is_coding(self) -> bool:
        """
        Check if the GTF record is protein-coding based on its attributes.

        Returns
        -------
        bool
            True if the record is protein-coding, False otherwise.
        """
        return self.attributes.get("gene_type", "") == "protein_coding"


class Collector(ABC):
    """
    Abstract base class for collecting features from a GTF file.
    
    Attributes
    ----------
    gtf : str
        Path to the GTF file.
    feature_type : str
        The type of feature to collect (e.g., 'transcript', 'exon', 'CDS').
    """
    
    def __init__(self, gtf: str, feature_type: str):
        self.gtf = gtf
        self.feature_type = feature_type
        self._features = self._collect_features()
    
    def _collect_features(self):
        """
        Parse the GTF file and collect features of the specified type.
        Calls the abstract _process_entry method for each matching entry.
        """
        features = self._initialize_container()
        with open(self.gtf, 'r') as gtf:
            for line in gtf:
                if not line.startswith("#"):
                    entry = GTF_record(*line.rstrip().split('\t'))
                    if entry.feature_type == self.feature_type:
                        self._process_entry(entry, features)
        return features
    
    @abstractmethod
    def _initialize_container(self):
        """Initialize the data structure to hold collected features."""
        pass
    
    @abstractmethod
    def _process_entry(self, entry: GTF_record, features):
        """Process a single GTF entry and add it to the features collection."""
        pass
    
    def __repr__(self):
        return f"{self.__class__.__name__} with {self._count_features()} {self.feature_type}s"
    
    @abstractmethod
    def _count_features(self) -> int:
        """Return the count of features for the repr method."""
        pass


class TranscriptCollector(Collector):
    """Collector for transcript features from a GTF file."""
    
    def __init__(self, gtf: str):
        super().__init__(gtf, feature_type='transcript')
    
    def _initialize_container(self):
        return defaultdict()
    
    def _process_entry(self, entry: GTF_record, features):
        transcript_id = entry.attributes.get('transcript_id')
        features[transcript_id] = (int(entry.start), int(entry.end))
    
    def _count_features(self) -> int:
        return len(self._features)
    
    def return_absolute_coords(self, tid: str, pos: Optional[Union[int, Iterable]]):
        """
        Convert relative positions to absolute genomic coordinates.
        
        Parameters
        ----------
        tid : str
            Transcript ID.
        pos : int or Iterable
            Position(s) relative to transcript start.
            
        Returns
        -------
        int, generator, or None
            Absolute coordinate(s) or None if transcript not found.
        """
        if isinstance(pos, int):
            positions = 1
        else:
            positions = len(pos)
        try:
            tid_start, tid_end = self._features[tid]
            if positions == 1:
                return tid_start + pos
            else:
                return (tid_start + j for j in pos)
        except KeyError:
            logging.info(f"Transcript {tid} not found in TranscriptCollector. Returning None")
            if positions == 1:
                return None
            else:
                return (None,) * positions


class ExonCollector(Collector):
    """Collector for exon features from a GTF file."""
    
    def __init__(self, gtf: str):
        super().__init__(gtf, feature_type='exon')
    
    def _initialize_container(self):
        return defaultdict(dict)
    
    def _process_entry(self, entry: GTF_record, features):
        transcript_id = entry.attributes.get('transcript_id')
        exon_id = entry.attributes.get('exon_id')
        features[transcript_id][exon_id] = (
            int(entry.start), int(entry.end), entry.strand
        )
    
    def _count_features(self) -> int:
        return len(self._features)

class CDSCollector(Collector):
    """Collector for CDS (coding sequence) features from a GTF file."""
    
    def __init__(self, gtf: str):
        super().__init__(gtf, feature_type='CDS')
    
    def _initialize_container(self):
        return defaultdict(dict)
    
    def _process_entry(self, entry: GTF_record, features):
        exon_id = entry.attributes.get('exon_id')
        # CDS entries might use 'protein_id' or just index by position
        cds_id = entry.attributes.get('exon_id', f"{entry.start}_{entry.end}")
        features[exon_id][cds_id] = (
            int(entry.start), int(entry.end), entry.strand
        )
    
    def compute_frame(self, exon_id: str, pos: Optional[Union[int, Iterable]]):
        """
        Compute the reading frame for a given position or positions within an exon.

        Parameters
        ----------
        exon_id : str
            Exon ID.
        pos : int or Iterable
            Position(s) relative to the exon start.

        Returns
        -------
        int, list, or None
            Reading frame(s) (0, 1, 2) or None if exon not found.
        """
        try:
            cds_start, _ = self._features[exon_id]
            if isinstance(pos, int):
                return (pos - cds_start) % 3
            else:
                return [(p - cds_start) % 3 for p in pos]
        except KeyError:
            logging.info(f"Exon {exon_id} not found in CDSCollector. Returning None")
            if isinstance(pos, int):
                return None
            else:
                return [None] * len(pos)
    
    def _count_features(self) -> int:
        return len(self._features)