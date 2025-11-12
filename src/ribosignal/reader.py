import pandas as pd
import ast
from typing import List
import logging
from collections import namedtuple
import itertools
import pysam
# NOTE: this dependencty should be avoided at all, as it's used just for reverse complement
from Bio.Seq import Seq


# The CodonTable is used to assign an index to each codon.
# All the stop codons will share the same index. 
bases = ['A', 'C', 'G', 'T']
stop_codons = ['TAA', 'TAG', 'TGA']
codons = [''.join(p) for p in itertools.product(bases, repeat=3) if ''.join(p) not in stop_codons]
codon_dict = {k:v for k,v in zip(codons + stop_codons, [i for i in range(1,61)] + [61]*3)}
CodonTable = namedtuple('CodonTable', codon_dict.keys())(**codon_dict)



class RibotishReader(object):
    """
    Class for handling predictions and profiles all together. 
    """
    def __init__(self, predictions: str, profile: str, genome: str):
        self._predictions = predictions # ORFs
        self._profile = profile 
        self._genome = pysam.FastaFile(genome)
        self.predictions = self._read_predictions()

    @staticmethod
    def _get_codons(chrom: str, start: int, end: int, strand: str,  genome: pysam.FastaFile):
        seq = genome.fetch(
            chrom, start, end
        )
        if strand == "-":
            seq = Seq(seq).reverse_complement()
        # fragment in codons. 
        codons = [seq[i:i+3] for i in range(0, len(seq), 3)]
        # return indexes, but with the same length of the input range. 
        codons = [getattr(CodonTable, codon)*3 for codon in codons]
        return codons

    def _read_predictions(self):
        predictions = pd.read_csv(self._predictions, sep='\t')
        profile = pd.read_csv(self._profile, sep='\t', converters={
            'TisProf': ast.literal_eval,
            'RiboProf': ast.literal_eval,
        }) # P profiles, that has coordinates relative to annotated TIS. 
        predictions['TisStart'] = predictions['GenomePos'].apply(lambda j: int(j.split(':')[1].split('-')[0]))
        predictions['profile'] = predictions['Tid'].map(profile.set_index("Tid")['RiboProf'])                
        # profile is a dict, but it has no entry for 0 positions. Relativize also the coordinates to be 1-based on the transcript
        predictions['profile'] = predictions.apply(lambda row: {(i-row['Start'])+1:row['profile'].get(i, 0) for i in range(row['Start'], row['Stop']+1)}, axis=1)
        predictions[["chrom", "range", "strand"]] = predictions["GenomePos"].str.split(":", n=2, expand=True)
        predictions[["start", "end"]] = predictions["range"].str.split("-", n=1, expand=True)
        predictions["start"] = predictions["start"].astype(int)
        predictions["end"] = predictions["end"].astype(int)
        # fetch the sequences in a codon based fashion.
        predictions["codons"] = predictions.apply(lambda row: self._get_codons(
            row["chrom"], row["start"], row["end"], row["strand"], self._genome), axis=1)
        # frame is easy, as the first position is always the start codon, thus the frame is 0. 
        # The returned array has [0,1,2,.., 0,1,2], matching the length of the ORF
        predictions["frame"] =  predictions["codons"].apply(
            lambda codons: list(range(3)) * (len(codons) // 3) + list(range(len(codons) % 3))
        )
        return predictions



class PredictionsReader(object):
    def __init__(self, predictions: str):
        self._prediction_file = predictions
        self._entries = self._read_predictions()
    
    def _read_predictions(self):
        predictions = pd.read_csv(self._prediction_file, sep='\t')
        # keep just those with `T`
        predictions = predictions.loc[predictions['RiboPStatus'] == 'T']
        # unpack into a more convenient item that could be queried
        predictions[["TisChrom", "TisPos", "TisStrand"]] = predictions["GenomePos"].str.split(':', n=2, expand=True)
        predictions[["TisStart", "TisEnd"]] = predictions["TisPos"].str.split("-", n=1, expand=True)
        predictions["TisStart"] = predictions["TisStart"].astype(int)
        predictions["TisEnd"] = predictions["TisEnd"].astype(int)
        predictions = predictions.sort_values(by=['TisStart', 'TisEnd'])
        entries = predictions.groupby('TisChrom').apply(
            lambda group: {
                col: group[col].to_numpy() for col in ["TisStart", "TisEnd", "TisStrand", "Tid", "Symbol"]
            }
        ).to_dict()
        return entries

    def _get_overlapping(self, chrom: str, pos: int) -> List[tuple]:
        try:
            # Retrieve the data for the specified chromosome
            chrom_data = self._entries[chrom]
            starts = chrom_data['TisStart']
            ends = chrom_data['TisEnd']
            
            # Find the rightmost interval that could potentially overlap
            # (intervals where start <= pos)
            right_idx = starts.searchsorted(pos, side='right')
            
            # Collect overlapping entries by checking backwards from right_idx
            overlapping = []
            for idx in range(right_idx - 1, -1, -1):
                # If start > pos, this interval and all previous ones can't overlap
                if starts[idx] > pos:
                    break
                # Check if pos falls within [start, end]
                if starts[idx] <= pos <= ends[idx]:
                    overlapping.append(
                        (starts[idx], ends[idx], chrom_data['TisStrand'][idx], 
                        chrom_data['Tid'][idx])
                    )
                # If end < pos, no earlier intervals can overlap (since sorted by start)
                elif ends[idx] < pos:
                    break    
            return overlapping
        except KeyError:
            logging.warning(f"Chromosome {chrom} not found in entries.")
            return []


class RibotishReader(object):
    """
    RibotishReader is a class designed to parse and store ribosome profiling data 
    from a given input file. It processes transcript-specific profiles and converts 
    them into absolute coordinates using a TranscriptCollector instance.

    Attributes:
        _profile (str): Path to the input file containing ribosome profiling data.
        _tc (TranscriptCollector): An instance of TranscriptCollector used for 
            coordinate transformations.
        tis_profiles (dict): A dictionary mapping transcript IDs to their TIS 
            (Translation Initiation Site) profiles.
        ribo_profiles (dict): A dictionary mapping transcript IDs to their ribosome 
            profiles.

    Methods:
        __init__(profile: str, tc: TranscriptCollector):
            Initializes the RibotishReader instance, parses the input file, and 
            stores the profiles.

        _parse_riboprofile(riboprof: str, tid: str, tc: TranscriptCollector) -> dict:
            Parses a ribosome profile string, converts relative positions to 
            absolute coordinates, and returns a dictionary representation of the 
            profile.

        _store_profile() -> Tuple[dict, dict]:
            Reads the input file, processes each line to extract TIS and ribosome 
            profiles, and stores them in dictionaries.
    """
    def __init__(self, profile: str, tc):
        self._profile = profile
        self._tc = tc
        self.tis_profiles, self.ribo_profiles = self._store_profile()

    @staticmethod
    def _parse_riboprofile(riboprof: str, tid: str, tc) -> dict:
        # drop {} and turn into a dict
        dict_profile = {}
        # Remove braces and strip whitespace
        cleaned = riboprof.replace("{", "").replace("}", "").strip()
        # If empty after cleaning, return empty dict
        if not cleaned:
            return None
        # Split by comma and process each entry
        fields = cleaned.split(",")
        for entry in fields:
            entry = entry.strip()  # Remove whitespace around each entry
            if not entry:  # Skip empty strings
                continue
            pos, v = entry.split(":")
            pos = pos.strip()
            v = v.strip()
            try:
                pos = tc.return_absolute_coords(tid, int(pos))
            except Exception as e:
                print(f"Error with pos={pos}, tid={tid}: {e}")
                raise
            dict_profile[pos] = int(v)  # Convert value to int if needed
        return dict_profile

    def _store_profile(self):
        tis_profiles = {}
        ribo_profiles = {}
        with open(self._profile, "r") as rprofile:
            for idx, line in enumerate(rprofile):
                if idx > 0:
                    _, tid, _, TisProf, RiboProf = line.rstrip().split("\t")
                    TisProf = self._parse_riboprofile(TisProf, tid, self._tc)
                    RiboProf = self._parse_riboprofile(RiboProf, tid, self._tc)
                    tis_profiles[tid] = TisProf
                    ribo_profiles[tid] = RiboProf
        return tis_profiles, ribo_profiles
    

class RibotishReaderWithCodons(RibotishReader):
    def __init__(self, tc):
        super(RibotishReader).__init__()
