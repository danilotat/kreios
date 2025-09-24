import torch
import math
from abc import ABC, abstractmethod
from region import Region, VCFRegion
from model import SimpleGenomicCNN
from genome import retrieve_sequence
import logging
import fetch
import numpy as np
import pysam


class ChannelEncoder(ABC):
    """
    Abstract base class for creating a specific tensor channel from a list of reads.
    """
    def __init__(self, name: str, dtype: torch.dtype, fill_value):
        self.name = name
        self.dtype = dtype
        self.fill_value = fill_value

    def create_empty_tensor(self, shape: tuple, device: str) -> torch.Tensor:
        """
        Instantiate the empty tensor
        """
        return torch.full(shape, fill_value=self.fill_value, dtype=self.dtype, device=device)

    @abstractmethod
    def _get_read_data(self, masked_read: fetch.MaskedRead) -> np.ndarray:
        """
        Abstract method to get the specific data array from a MaskedRead.
        This is replaced within each subclass according to the channel's logic
        """
        pass

    def __call__(self, reads_to_process: list[fetch.MaskedRead], start_row: int, tensor_shape: tuple, region_start: int, device: str, **kwargs) -> torch.Tensor:
        """
        Processes a list of reads to generate the final tensor for this channel.
        
        This method contains the logic for placing individual read data into the
        correct location within the final, centered tensor.
        """
        final_tensor = self.create_empty_tensor(tensor_shape, device)
        region_length = tensor_shape[1]

        for i, m_read in enumerate(reads_to_process):
            # Calculate horizontal placement (same logic as before)
            tensor_start = max(0, m_read.start - region_start)
            tensor_end = min(region_length, m_read.end - region_start)
            if tensor_start >= tensor_end: continue

            # Calculate the slice of the read's data to use
            read_slice_start = max(0, region_start - m_read.start)
            read_slice_end = read_slice_start + (tensor_end - tensor_start)
            
            # The destination row in the tensor is offset by start_row for centering
            dest_row = start_row + i

            # Get the specific data using the subclass implementation
            padded_array = self._get_read_data(m_read)
            data_to_place = padded_array[read_slice_start:read_slice_end]
            
            # Place the data into the final tensor
            final_tensor[dest_row, tensor_start:tensor_end] = torch.from_numpy(data_to_place).to(self.dtype)
            
        return final_tensor


class SequenceEncoder(ChannelEncoder):
    def __init__(self, base_color_stride=20, offset_t_c=5, offset_a_g=4, default_color=0):
        # Define the color mapping for each base. This concept is inspired by the read base channel in DeepVariant.
        self.BASE_COLOR_MAP = {
            'A': offset_a_g + base_color_stride * 3,
            'G': offset_a_g + base_color_stride * 2,
            'T': offset_t_c + base_color_stride * 1,
            'C': offset_t_c + base_color_stride * 0,
            'N': default_color,
            '-': default_color 
        }
        
        # Ensure values are within uint8 range
        for base, value in self.BASE_COLOR_MAP.items():
            if not (0 <= value <= 255):
                raise ValueError(f"Color for base '{base}' is {value}, which is outside the [0, 255] range.")

        super().__init__(
            name='sequence_color',
            dtype=torch.uint8, 
            fill_value=default_color
        )
        
        self.vectorized_mapper = np.vectorize(
            lambda base: self.BASE_COLOR_MAP.get(base.upper(), default_color)
        )

    def _get_read_data(self, masked_read: fetch.MaskedRead) -> np.ndarray:
        """
        Gets the sequence data and applies the BaseColor mapping to convert
        characters to integer "colors".
        """
        sequence_chars = masked_read.get_padded_array('sequence')
        return self.vectorized_mapper(sequence_chars)


class BaseQualityEncoder(ChannelEncoder):
    """Encodes the base quality score channel."""
    def __init__(self, padding_value=np.nan):
        super().__init__(
            name='base_qualities', 
            dtype=torch.float32, 
            fill_value=padding_value
        )
        # base above this value will be capped 
        self.base_quality_cap = 16

    def _get_read_data(self, masked_read: fetch.MaskedRead) -> np.ndarray:
        base_qualities = masked_read.get_padded_array('base_qualities')
        # Cap the base qualities to avoid extreme values
        np.clip(base_qualities, a_min=0, a_max=self.base_quality_cap, out=base_qualities)
        return base_qualities

class IntronicEncoder(ChannelEncoder):
    """Encodes whether the read base is in an intronic region."""
    def __init__(self, intron_value=1, non_intron_value=0, padding_value=-1):
        super().__init__(
            name='intronic', 
            dtype=torch.int8, 
            fill_value=padding_value
        )
        self.intron_value = intron_value
        self.non_intron_value = non_intron_value

    def _get_read_data(self, masked_read: fetch.MaskedRead) -> np.ndarray:
        # For simplicity, let's assume that intronic regions are marked by 'N' in the sequence.
        read_sequence = masked_read.get_padded_array('sequence').astype(str)
        intron_array = np.full(len(read_sequence), fill_value=self.fill_value, dtype=np.int8)
        for i, base in enumerate(read_sequence):
            if base == 'N':
                intron_array[i] = self.intron_value
            elif base == '-':
                continue  # Skip padding
            else:
                intron_array[i] = self.non_intron_value
        return intron_array

class ReferenceMatchingEncoder(ChannelEncoder):
    """Encodes whether the read base matches the reference base."""
    def __init__(self, padding_value=-1):
        super().__init__(
            name='ref_match', 
            dtype=torch.int8, 
            fill_value=padding_value
        )
        self.reference_matching = 11
        self.reference_mismatching = 16
        
        # These will be set by the __call__ method for each region
        self.reference_sequence = None
        self.region_start = None

    def __call__(self, reads_to_process: list, start_row: int, tensor_shape: tuple, region_start: int, device: str, **kwargs) -> torch.Tensor:
        """
        Overrides the parent __call__ to capture the 'reference_sequence' 
        and 'region_start' context before processing.
        """
        ref_seq = kwargs.pop('reference_sequence', None)
        if ref_seq is None:
            raise ValueError("ReferenceMatchingEncoder requires 'reference_sequence' to be passed via kwargs.")
        
        # Set the context for this specific call
        self.reference_sequence = ref_seq.upper()
        self.region_start = region_start
        
        # Now, call the parent's __call__ method to execute the main loop
        return super().__call__(
            reads_to_process, 
            start_row, 
            tensor_shape, 
            region_start, 
            device, 
            **kwargs
        )

    def _get_read_data(self, masked_read: fetch.MaskedRead) -> np.ndarray:
        """
        Gets the match/mismatch data using the context set by __call__.
        This version correctly maps global read coordinates to the local
        reference sequence coordinates.
        """
        if self.reference_sequence is None or self.region_start is None:
            raise RuntimeError("The reference sequence context has not been set. Ensure this encoder is called correctly.")

        read_sequence = masked_read.get_padded_array('sequence').astype(str)
        read_len = len(read_sequence)
        ref_len = len(self.reference_sequence)
        
        match_array = np.full(read_len, fill_value=self.fill_value, dtype=np.int8)

        # Iterate through each base of the read
        for i in range(read_len):
            read_char = read_sequence[i]
            if read_char == '-':
                continue  # Skip padding characters in the read

            # 1. Calculate the base's global genomic position
            genomic_pos = masked_read.start + i
            
            # 2. Convert the global position to an index in our reference_sequence
            ref_idx = genomic_pos - self.region_start
            
            # 3. Check if this position is valid within our fetched reference
            if 0 <= ref_idx < ref_len:
                ref_char = self.reference_sequence[ref_idx]
                
                if read_char == ref_char:
                    match_array[i] = self.reference_matching
                else:
                    match_array[i] = self.reference_mismatching
                    
        return match_array


class ReadStrandEncoder(ChannelEncoder):
    """
    Encodes reads based on whether they map to a strand or another.
    """
    def __init__(self, padding_value=-1):
        super().__init__(
            name='read_strand',  # A more descriptive name
            dtype=torch.int8, 
            fill_value=padding_value
        )
        self.positive_strand_color = 12
        self.negative_strand_color = 15

    def _get_read_data(self, masked_read: fetch.MaskedRead) -> np.ndarray:
        # This base-level method is not used by this encoder's logic.
        # The logic is handled entirely within the overridden __call__.
        raise NotImplementedError("ReadStrandEncoder does not use _get_read_data.")

    def __call__(self, reads_to_process: list[fetch.MaskedRead], start_row: int, tensor_shape: tuple, region_start: int, device: str, **kwargs) -> torch.Tensor:
        """
        Processes a list of reads to generate a tensor showing read strand.
        """
        final_tensor = self.create_empty_tensor(tensor_shape, device)
        region_length = tensor_shape[1]
        for i, m_read in enumerate(reads_to_process):
            color = self.positive_strand_color
            if m_read.read.is_reverse:
                color = self.negative_strand_color
            # computing of read placement 
            tensor_start = max(0, m_read.start - region_start)
            tensor_end = min(region_length, m_read.end - region_start)
            if tensor_start >= tensor_end: 
                continue
            dest_row = start_row + i
            # Fill the segment of the tensor corresponding to the read
            final_tensor[dest_row, tensor_start:tensor_end] = color
        return final_tensor


class ReadsSupportingAlleleEncoder(ChannelEncoder):
    """
    Encodes reads based on whether they support a given alternate allele.
    
    Each read is assigned one of two values:
    - `allele_supporting_read`: If the read's sequence contains the alternate allele
      at the specified variant position.
    - `allele_unsupporting_read`: Otherwise.
      
    This value is tiled across the entire span of the read in the final tensor.
    """
    def __init__(self, padding_value=-1):
        super().__init__(
            name='allele_support',  # A more descriptive name
            dtype=torch.int8, 
            fill_value=padding_value
        )
        self.allele_supporting_read = 2
        self.allele_unsupporting_read = 10

    def _get_read_data(self, masked_read: fetch.MaskedRead) -> np.ndarray:
        # This base-level method is not used by this encoder's logic.
        # The logic is handled entirely within the overridden __call__.
        raise NotImplementedError("ReadsSupportingAlleleEncoder does not use _get_read_data.")

    def __call__(self, reads_to_process: list[fetch.MaskedRead], start_row: int, tensor_shape: tuple, region_start: int, device: str, **kwargs) -> torch.Tensor:
        final_tensor = self.create_empty_tensor(tensor_shape, device)
        region_length = tensor_shape[1]
        variant = kwargs.get('variant')
        if not isinstance(variant, VCFRegion):
            raise ValueError(
                "ReadsSupportingAlleleEncoder requires a 'variant' of type VCFRegion "
                "to be passed as a keyword argument."
            )
            
        variant_pos = variant.start
        alt_allele = variant.alt
        for i, m_read in enumerate(reads_to_process):
            supports_alt = False            
            try:
                ref_positions = m_read.read.get_reference_positions(full_length=True)
                query_index = ref_positions.index(variant_pos)
                read_base = m_read.read.get_forward_sequence()[query_index].upper()
                if read_base == alt_allele:
                    supports_alt = True
            except (ValueError, IndexError):
                # This occurs if the read doesn't span the variant position.
                # In this case, supports_alt correctly remains False.
                pass

            # 4. Determine the color and fill the tensor for this read
            color = self.allele_supporting_read if supports_alt else self.allele_unsupporting_read
            tensor_start = max(0, m_read.start - region_start)
            tensor_end = min(region_length, m_read.end - region_start)
            
            if tensor_start >= tensor_end: 
                continue
                
            dest_row = start_row + i
            
            # Fill the entire segment of the tensor corresponding to the read
            final_tensor[dest_row, tensor_start:tensor_end] = color
            
        return final_tensor


class AlleleFrequencyEncoder(ChannelEncoder):
    """
    Encodes a channel where the color of an entire read is determined by the
    allele frequency of the specific variant it supports.
    """
    K_MAX_PIXEL_VALUE_AS_FLOAT = 255.0

    def __init__(self, min_non_zero_allele_frequency: float = 0.01):
        if not (0 < min_non_zero_allele_frequency < 1):
            raise ValueError("min_non_zero_allele_frequency must be between 0 and 1.")
        
        self.min_non_zero_af = min_non_zero_allele_frequency
        self.log10_min_af = math.log10(self.min_non_zero_af)
        # The fill value for empty regions is the color for AF=0
        fill_color = self._allele_frequency_color(0.0)
        super().__init__(
            name='allele_frequency', 
            dtype=torch.uint8,
            fill_value=fill_color
        )
    
    def _get_read_data(self, masked_read: fetch.MaskedRead) -> np.ndarray:
        # This method is not used for this encoder, as data is not derived per-base.
        raise NotImplementedError("AlleleFrequencyEncoder does not use _get_read_data.")

    def _allele_frequency_color(self, allele_frequency: float) -> int:
        """Converts a float allele frequency to a uint8 color value."""
        if allele_frequency <= self.min_non_zero_af:
            return 0
        else:
            log10_af = math.log10(allele_frequency)
            # This formula scales the log-frequency to the [0, 255] range.
            scaled_value = ((self.log10_min_af - log10_af) / self.log10_min_af) * self.K_MAX_PIXEL_VALUE_AS_FLOAT
            return min(255, int(scaled_value))

    def _get_read_allele_frequency(self, read: pysam.AlignedSegment, data: fetch.VariantCallData) -> float:
        """Finds which alt allele the read supports and returns its frequency."""
        # In BAM files, query_name is the unique identifier for a read/fragment.
        read_key = read.query_name
        
        # Iterate over the alts that have supporting read information.
        for alt_allele, supporting_reads in data.allele_support.items():
            # Check if this allele is one we are currently considering.
            if alt_allele not in data.alt_alleles:
                continue
            
            # Check if this read is in the list of supporters for this allele.
            if read_key in supporting_reads:
                # If it is, return the allele's frequency.
                return data.allele_frequencies.get(alt_allele, 0.0)
        
        # If the read doesn't support any of the considered alts, return 0.
        return 0.0

    # Override the entire __call__ method for custom logic
    def __call__(self, reads_to_process: list[fetch.MaskedRead], start_row: int, 
                 tensor_shape: tuple, region_start: int, device: str, **kwargs) -> torch.Tensor:
        
        variant_data = kwargs.get('variant_call_data')
        if not variant_data:
            logging.warning("AlleleFrequencyEncoder requires 'variant_call_data' but none was provided. Returning empty tensor.")
            return self.create_empty_tensor(tensor_shape, device)

        final_tensor = self.create_empty_tensor(tensor_shape, device)
        region_length = tensor_shape[1]

        for i, m_read in enumerate(reads_to_process):
            # 1. Get the allele frequency for this specific read
            read_af = self._get_read_allele_frequency(m_read.read, variant_data)
            
            # 2. Convert that frequency to a single color
            color = self._allele_frequency_color(read_af)

            # 3. Apply that color across the entire length of the read in the tensor
            tensor_start = max(0, m_read.start - region_start)
            tensor_end = min(region_length, m_read.end - region_start)
            if tensor_start >= tensor_end: continue
            
            dest_row = start_row + i
            final_tensor[dest_row, tensor_start:tensor_end] = color
            
        return final_tensor


if __name__ == "__main__":
    from plotter import TensorVisualizer
    region = 'chr5:140561643-140561644'
    variant = VCFRegion(
        region, ref='A', alt='T'
    )
    simplecnn = SimpleGenomicCNN()
    # extend the region by 50 bp 
    region = 'chr5:140561593-140561694'
    region_object = Region(region)
    genome = '/CTGlab/db/GRCh38_GIABv3_no_alt_analysis_set_maskedGRC_decoys_MAP2K3_KMT2C_KCNJ18.fasta'
    reference_sequence = retrieve_sequence(genome, region_object)
    print(f"Length of reference_sequence is {len(reference_sequence)}")
    all_encoders = [
        BaseQualityEncoder(),
        SequenceEncoder(),
        IntronicEncoder(),
        ReferenceMatchingEncoder(),
        ReadsSupportingAlleleEncoder(),
        ReadStrandEncoder()

    ]
    bam = "/CTGlab/projects/ribo/goyal_2023/RNAseq/SRR20649710_GSM6395082.markdup.sorted.bam"
    fetcher = fetch.ReadFetcher(bam, max_reads=256, channel_encoders=all_encoders, genome=genome, device='cpu')
# 5. Call fetch_tensors and pass the variant-specific data.
    chromosome, positions = region.split(':')
    start, end = map(int, positions.split('-'))
    tensors = fetcher.fetch_tensors(chromosome, start, end, variant)
    single_tensor = torch.nan_to_num(
        torch.stack(list(tensors.values()), dim=0))
    print(simplecnn(single_tensor.unsqueeze(0)))
    fig = TensorVisualizer().plot(tensors, chromosome, start, end)
    fig.savefig("test_encoders.png")
