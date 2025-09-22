import torch
import math
from abc import ABC, abstractmethod
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
        """Creates a tensor full of the specified padding/fill value."""
        return torch.full(shape, fill_value=self.fill_value, dtype=self.dtype, device=device)

    @abstractmethod
    def _get_read_data(self, masked_read: fetch.MaskedRead) -> np.ndarray:
        """Abstract method to get the specific data array from a MaskedRead."""
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
    def __init__(self, base_color_stride=6, offset_t_c=5, offset_a_g=4, default_color=0):
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
    def __init__(self, reference_sequence: str, padding_value=-1):
        super().__init__(
            name='ref_match', 
            dtype=torch.int8, 
            fill_value=padding_value
        )
        self.reference_matching = 11
        self.reference_mismatching = 12
        self.reference_sequence = reference_sequence.upper()

    def _get_read_data(self, masked_read: fetch.MaskedRead) -> np.ndarray:
        read_sequence = masked_read.get_padded_array('sequence').astype(str)
        ref_length = len(self.reference_sequence)
        read_length = len(read_sequence)
        # instantiate the match array with the fill value
        match_array = np.full(read_length, fill_value=self.fill_value, dtype=np.int8)
        # Determine the overlap between the read and the reference
        overlap_start = max(0, masked_read.start)
        overlap_end = min(ref_length, masked_read.end)
        if overlap_start < overlap_end:
            ref_segment = self.reference_sequence[overlap_start:overlap_end]
            read_segment = read_sequence[overlap_start - masked_read.start : overlap_end - masked_read.start]
            for i in range(len(read_segment)):
                if read_segment[i] == '-':
                    continue  # Skip padding
                elif read_segment[i] == ref_segment[i]:
                    match_array[overlap_start - masked_read.start + i] = self.reference_matching
                else:
                    match_array[overlap_start - masked_read.start + i] = self.reference_mismatching
        return match_array

class AlleleFrequencyEncoder(ChannelEncoder):
    """
    Encodes a channel where the color of an entire read is determined by the
    allele frequency of the specific variant it supports.
    #@TODO: still a WIP
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
    all_encoders = [
        BaseQualityEncoder(),
        SequenceEncoder(),
        IntronicEncoder(),
    ]
    pos = 'chr5:140561643-140561644'
    region = 'chr5:140561543-140561743'
    bam = "/CTGlab/projects/ribo/goyal_2023/RNAseq/SRR20649710_GSM6395082.markdup.sorted.bam"
    fetcher = fetch.ReadFetcher(bam, max_reads=256, channel_encoders=all_encoders, device='cpu')
# 5. Call fetch_tensors and pass the variant-specific data.
    chromosome, positions = region.split(':')
    start, end = map(int, positions.split('-'))
    tensors = fetcher.fetch_tensors(chromosome, start, end)
    print(tensors.keys())
    print(tensors['sequence_color'].shape, tensors['base_qualities'].shape, tensors['intronic'].shape)
    fig = TensorVisualizer().plot(tensors, chromosome, start, end)
    fig.savefig("test_encoders.png")

# Now final_tensors['allele_frequency'] will be a tensor where:
# - Rows for reads 'read_name_1', 'read_name_3', 'read_name_8' are colored with the value for AF=0.25.
# - Rows for other reads (like 'read_name_5') are colored with the value for AF=0.
# - Empty rows are also colored with the value for AF=0.

# print(final_tensors.keys())
# Expected: dict_keys(['sequence_color', 'base_qualities', 'allele_frequency'])